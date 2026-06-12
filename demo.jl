using Pkg
Pkg.activate(@__DIR__)

using RiverSiteEvolution
using Plots
using Printf
using Random

println("="^60)
println("河流-遗址演化模型 演示脚本")
println("River-Site Evolution Model Demo")
println("="^60)

gr()
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

Random.seed!(42)

println("\n[1/6] 设置计算域和参数...")
Nx = 50
Ny = 40
Lx = 1000.0
Ly = 800.0

domain_base = ShallowWaterDomain(Nx, Ny, Lx, Ly)

sw_params = ShallowWaterParams(
    g=9.81,
    n=0.03,
    ν=1.0,
    h₀=1e-3,
    dt_max=0.1
)

println("    计算域: $(Nx)×$(Ny) 网格, $(Lx)×$(Ly) m")
println("    空间步长: dx=$(domain_base.dx)m, dy=$(domain_base.dy)m")

println("\n[2/6] 生成考古遗址数据...")
n_sites = 8
sites = generate_synthetic_sites(
    Lx, Ly, n_sites;
    min_elevation=1.5,
    max_elevation=4.0,
    min_radius=25.0,
    max_radius=70.0,
    min_age=2000.0,
    max_age=8000.0,
    seed=123
)

for site in sites
    @printf("    %-10s | 位置: (%6.1f, %6.1f) | 高程: %.2fm | 半径: %.1fm | 年代: %.0f-%.0f BP | 文化: %s\n",
        site.id, site.x, site.y, site.elevation, site.radius,
        site.start_age, site.end_age, site.culture)
end

p_sites = plot_site_locations(sites, domain_base;
    title="考古遗址分布与年代序列")
savefig(p_sites, joinpath(output_dir, "01_site_locations.png"))
println("    ✓ 遗址分布图已保存: 01_site_locations.png")

println("\n[3/6] 构建含遗址地形的计算域并求解浅水方程...")

true_paleo = PaleoriverParams(
    inflow_rate=1.5,
    river_center_y=Ly/2,
    river_width=120.0,
    river_depth=2.5,
    roughness=0.035,
    sediment_flux=0.0
)

zb_with_sites = create_topography_with_sites(
    domain_base.x, domain_base.y, sites;
    base_slope=0.0015,
    river_channel_depth=true_paleo.river_depth,
    river_width=true_paleo.river_width,
    river_center_y=true_paleo.river_center_y
)

domain = ShallowWaterDomain(Nx, Ny, Lx, Ly, zb_with_sites)

p_topo = plot_topography(domain, sites;
    title="地形高程图（含遗址抬高）")
savefig(p_topo, joinpath(output_dir, "02_topography.png"))
println("    ✓ 地形图已保存: 02_topography.png")

tspan = (0.0, 50.0)
eval_times = collect(range(5.0, 50.0, length=10))

sol = solve_shallow_water(
    domain, sw_params, tspan;
    inflow_rate=true_paleo.inflow_rate,
    saveat=eval_times
)

println("    PDE 求解完成, 时间区间: $tspan")
println("    保存时间点数量: $(length(sol.sol.t))")

state_final = sol(tspan[2])

p_flow = plot_flow_field(state_final, domain, sites;
    title="流场分布 (t=$(tspan[2]))",
    quiver_scale=0.05,
    quiver_density=4)
savefig(p_flow, joinpath(output_dir, "03_flow_field.png"))
println("    ✓ 流场图已保存: 03_flow_field.png")

p_risk = plot_inundation_risk(state_final, domain, sites;
    title="遗址淹没风险图 (t=$(tspan[2]))",
    depth_threshold=0.5)
savefig(p_risk, joinpath(output_dir, "04_inundation_risk.png"))
println("    ✓ 淹没风险图已保存: 04_inundation_risk.png")

println("\n[4/6] 生成遗址淹没风险时间演化图...")

t_span_years = (minimum([s.start_age for s in sites]),
                maximum([s.end_age for s in sites]))
eval_times_full = collect(range(t_span_years[1], t_span_years[2], length=15))

p_evo = plot_site_evolution(sol, domain, sites, eval_times_full;
    title="遗址淹没风险时空演化热力图")
savefig(p_evo, joinpath(output_dir, "05_site_evolution.png"))
println("    ✓ 演化热力图已保存: 05_site_evolution.png")

println("\n[5/6] 参数反演：基于观测的古河道参数估计...")

println("    生成观测数据（含噪声）...")
obs_tspan = (0.0, 50.0)
obs_times = collect(range(5.0, 50.0, length=8))

noisy_obs, true_obs, sol_true, domain_true = generate_observations(
    true_paleo, domain_base, sites, obs_tspan, obs_times;
    noise_level=0.15
)

@printf("    观测数据维度: %d×%d (时间点×遗址数)\n", size(noisy_obs)...)
@printf("    真实参数: inflow=%.2f, center_y=%.1f, width=%.1f, depth=%.2f, n=%.4f\n",
    true_paleo.inflow_rate, true_paleo.river_center_y,
    true_paleo.river_width, true_paleo.river_depth, true_paleo.roughness)

param_names = ["inflow_rate", "river_center_y", "river_width", "river_depth", "roughness"]
lower_bounds = [0.5, Ly*0.2, 50.0, 1.0, 0.01]
upper_bounds = [3.0, Ly*0.8, 200.0, 5.0, 0.08]
initial_params = [1.0, Ly/2, 100.0, 2.0, 0.03]

inv_params = InversionParams(
    param_names, lower_bounds, upper_bounds, initial_params;
    max_iterations=30,
    tolerance=1e-4,
    algorithm=:LBFGS
)

println("    开始参数反演 (最多 $(inv_params.max_iterations) 迭代)...")
inv_result = invert_paleoriver_params(
    inv_params, domain_base, sites, obs_tspan, obs_times, noisy_obs;
    verbose=true
)

println("\n    反演结果:")
for (i, name) in enumerate(param_names)
    true_val = struct_to_param_vec(true_paleo, [name])[1]
    opt_val = inv_result.optimized_params[i]
    rel_err = abs(opt_val - true_val) / abs(true_val) * 100
    @printf("      %-16s: 真值=%.4f, 反演=%.4f, 相对误差=%.2f%%\n",
        name, true_val, opt_val, rel_err)
end
@printf("    收敛状态: %s, 迭代次数: %d, 最小目标值: %.6f\n",
    inv_result.converged, inv_result.iterations, inv_result.minimum_objective)

p_conv = plot_inversion_convergence(inv_result;
    title="参数反演收敛过程")
savefig(p_conv, joinpath(output_dir, "06_inversion_convergence.png"))
println("    ✓ 收敛曲线已保存: 06_inversion_convergence.png")

println("    计算参数灵敏度...")
sensitivity = compute_param_sensitivity(
    inv_result, inv_params, domain_base, sites,
    obs_tspan, obs_times, noisy_obs;
    epsilon=1e-3
)

p_sens = plot_param_sensitivity(inv_result, sensitivity;
    title="古河道参数灵敏度分析")
savefig(p_sens, joinpath(output_dir, "07_param_sensitivity.png"))
println("    ✓ 参数灵敏度图已保存: 07_param_sensitivity.png")

println("\n[6/6] 生成淹没过程动画...")
println("    正在渲染动画（可能需要几分钟）...")

try
    anim = animate_inundation(
        sol, domain, sites,
        joinpath(output_dir, "08_inundation_animation.gif");
        t_start=0.0,
        t_end=50.0,
        n_frames=30,
        fps=5,
        depth_threshold=0.5
    )
    println("    ✓ 动画已保存: 08_inundation_animation.gif")
catch e
    println("    ⚠ 动画生成跳过: $e")
end

println("\n" * "="^60)
println("演示完成！所有输出文件保存在: $output_dir")
println("="^60)
println("\n生成的文件列表:")
for f in sort(readdir(output_dir))
    fpath = joinpath(output_dir, f)
    sz = filesize(fpath) / 1024
    @printf("  - %-40s  (%6.1f KB)\n", f, sz)
end
println("\n模块架构:")
println("  1. PDESolver.jl         - 浅水方程PDE求解器 (DifferentialEquations.jl)")
println("  2. SiteCoupling.jl      - 遗址耦合模块 (地形抬高、边界条件)")
println("  3. ParameterInversion.jl - 参数反演模块 (Optim.jl + LBFGS)")
println("  4. Visualization.jl     - 可视化模块 (Plots.jl)")
