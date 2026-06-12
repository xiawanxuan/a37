module Visualization

using Plots
using Colors
using Statistics
using Printf

import ..PDESolver:
    ShallowWaterParams,
    ShallowWaterDomain,
    ShallowWaterState,
    ShallowWaterSolution,
    compute_inundation_depth,
    compute_flow_velocity

import ..SiteCoupling:
    ArchaeologicalSite,
    SiteChronology,
    compute_inundation_risk_map,
    compute_site_inundation_risk,
    get_active_sites

import ..ParameterInversion:
    PaleoriverParams,
    InversionResult,
    InversionParams

export
    plot_flow_field,
    plot_inundation_risk,
    plot_site_evolution,
    plot_inversion_convergence,
    plot_param_sensitivity,
    animate_inundation,
    plot_topography,
    plot_site_locations

function plot_topography(
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite}=ArchaeologicalSite[];
    title::String="Topography with Archaeological Sites",
    cmap::Symbol=:terrain,
    show_sites::Bool=true
)
    p = heatmap(domain.x[:, 1], domain.y[1, :], domain.zb',
                xlabel="X (m)", ylabel="Y (m)",
                title=title,
                color=cmap,
                colorbar_title="Elevation (m)",
                aspect_ratio=:equal)

    if show_sites && !isempty(sites)
        for site in sites
            scatter!(p, [site.x], [site.y],
                    markersize=max(4, site.radius / 10),
                    markeralpha=0.7,
                    markerstrokewidth=2,
                    label=site.id)
        end
    end

    return p
end

function plot_flow_field(
    state::ShallowWaterState,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite}=ArchaeologicalSite[];
    title::String="Flow Field",
    cmap::Symbol=:viridis,
    quiver_scale::Float64=0.1,
    quiver_density::Int=5
)
    Nx, Ny = domain.Nx, domain.Ny
    velocity = compute_flow_velocity(state)

    p1 = heatmap(domain.x[:, 1], domain.y[1, :], velocity',
                 xlabel="X (m)", ylabel="Y (m)",
                 title="Flow Velocity Magnitude",
                 color=cmap,
                 colorbar_title="Velocity (m/s)",
                 aspect_ratio=:equal)

    for site in sites
        scatter!(p1, [site.x], [site.y],
                markersize=max(4, site.radius / 10),
                marker=:circle,
                markeralpha=0.8,
                markercolor=:red,
                label=site.id)
    end

    skip_i = max(1, Nx ÷ quiver_density)
    skip_j = max(1, Ny ÷ quiver_density)

    x_q = domain.x[1:skip_i:end, 1:skip_j:end]
    y_q = domain.y[1:skip_i:end, 1:skip_j:end]
    u_q = state.u[1:skip_i:end, 1:skip_j:end] * quiver_scale
    v_q = state.v[1:skip_i:end, 1:skip_j:end] * quiver_scale

    p2 = quiver(vec(x_q), vec(y_q),
                quiver=(vec(u_q), vec(v_q)),
                xlabel="X (m)", ylabel="Y (m)",
                title="Flow Direction",
                arrow=arrow(:open, :head, 0.3, 0.3),
                aspect_ratio=:equal,
                linewidth=1.5,
                color=:blue)

    for site in sites
        scatter!(p2, [site.x], [site.y],
                markersize=max(4, site.radius / 10),
                marker=:circle,
                markeralpha=0.8,
                markercolor=:red,
                label="")
    end

    l = @layout [a; b]
    p = plot(p1, p2, layout=l, size=(800, 800), plot_title=title)

    return p
end

function plot_inundation_risk(
    state::ShallowWaterState,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite};
    title::String="Inundation Risk Map",
    cmap::Symbol=:YlOrRd,
    depth_threshold::Float64=0.5
)
    inundation = compute_inundation_depth(state, domain)
    risk_map = compute_inundation_risk_map(sites, inundation, domain.x, domain.y;
                                           depth_threshold=depth_threshold)

    p = heatmap(domain.x[:, 1], domain.y[1, :], risk_map',
                xlabel="X (m)", ylabel="Y (m)",
                title=title,
                color=cmap,
                colorbar_title="Inundation Risk",
                aspect_ratio=:equal,
                clims=(0, max(1, maximum(risk_map))))

    for site in sites
        site_risk = compute_site_inundation_risk(site, inundation, domain.x, domain.y;
                                                 depth_threshold=depth_threshold)
        color = site_risk > 0.5 ? :red : (site_risk > 0.2 ? :orange : :green)
        scatter!(p, [site.x], [site.y],
                markersize=max(6, site.radius / 8),
                marker=:circle,
                markeralpha=0.8,
                markercolor=color,
                markerstrokecolor=:black,
                markerstrokewidth=2,
                label=@sprintf("%s (risk=%.2f)", site.id, site_risk))
    end

    return p
end

function plot_site_evolution(
    sol::ShallowWaterSolution,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    eval_times::Vector{Float64};
    chrono_times::Union{Vector{Float64}, Nothing}=nothing,
    title::String="Site Inundation Risk Evolution",
    cmap::Symbol=:plasma
)
    n_times = length(eval_times)
    n_sites = length(sites)

    if chrono_times === nothing
        chrono_times = Float64[]
        for site in sites
            push!(chrono_times, site.start_age)
            push!(chrono_times, site.end_age)
        end
        chrono_min = minimum(chrono_times)
        chrono_max = maximum(chrono_times)
        chrono_times = collect(range(chrono_min, chrono_max, length=n_times))
    end

    @assert length(chrono_times) == n_times "chrono_times length must match eval_times length"

    risk_matrix = zeros(n_times, n_sites)

    for (t_idx, t) in enumerate(eval_times)
        state = sol(t)
        inundation = compute_inundation_depth(state, domain)

        for (s_idx, site) in enumerate(sites)
            risk = compute_site_inundation_risk(site, inundation, domain.x, domain.y)
            risk_matrix[t_idx, s_idx] = risk
        end
    end

    p1 = heatmap(chrono_times, 1:n_sites, risk_matrix',
                 xlabel="Time (years BP)",
                 ylabel="Sites",
                 yticks=(1:n_sites, [s.id for s in sites]),
                 title="Inundation Risk Heatmap",
                 color=cmap,
                 colorbar_title="Risk",
                 aspect_ratio=:auto,
                 size=(800, 400))

    p2 = plot(chrono_times, risk_matrix,
              xlabel="Time (years BP)",
              ylabel="Inundation Risk",
              title="Risk Time Series",
              label=hcat([s.id for s in sites]...),
              linewidth=2,
              legend=:outertopright,
              size=(800, 300))

    for (s_idx, site) in enumerate(sites)
        vspan!(p2, [site.start_age, site.end_age],
               alpha=0.1, color=:blue, label="")
    end

    l = @layout [a{0.6h}; b{0.4h}]
    p = plot(p1, p2, layout=l, size=(900, 700), plot_title=title)

    return p
end

function plot_inversion_convergence(
    result::InversionResult;
    title::String="Inversion Convergence",
    show_params::Bool=true
)
    n_iter = length(result.trace)

    p1 = plot(1:n_iter, result.trace,
              xlabel="Iteration",
              ylabel="Objective Value",
              title="Convergence Trace",
              color=:blue,
              linewidth=2,
              marker=:circle,
              markersize=3,
              yaxis=:log,
              legend=false)

    if show_params && size(result.param_history, 2) > 1
        n_params = length(result.param_names)
        colors = distinguishable_colors(n_params, [RGB(1,1,1), RGB(0,0,0)])

        p2 = plot(title="Parameter Evolution",
                  xlabel="Iteration",
                  ylabel="Parameter Value",
                  legend=:outertopright)

        for i in 1:n_params
            plot!(p2, 1:size(result.param_history, 2),
                  result.param_history[i, :],
                  label=result.param_names[i],
                  color=colors[i],
                  linewidth=2,
                  marker=:circle,
                  markersize=2)
        end
    else
        p2 = plot(title="Optimized Parameters",
                  framestyle=:none,
                  legend=false)
        for (i, (name, val)) in enumerate(zip(result.param_names, result.optimized_params))
            annotate!(p2, 0.5, (n_params - i + 0.5) / (n_params + 1),
                      text(@sprintf("%s = %.4f", name, val), 12, :left))
        end
    end

    n_params = length(result.param_names)
    status_text = @sprintf("Converged: %s\nIterations: %d\nMinimum: %.6f",
                          result.converged, result.iterations, result.minimum_objective)

    p3 = plot(title="Inversion Status",
              framestyle=:none,
              legend=false)
    annotate!(p3, 0.5, 0.5, text(status_text, 12, :center))

    l = @layout [a{0.6w} b{0.4w}; c{1.0w}]
    p = plot(p1, p2, p3, layout=l, size=(1000, 600), plot_title=title)

    return p
end

function plot_param_sensitivity(
    result::InversionResult,
    sensitivity::Vector{Float64};
    title::String="Parameter Sensitivity Analysis"
)
    n_params = length(result.param_names)

    colors = [s > 0 ? :red : :blue for s in sensitivity]

    p = bar(result.param_names, abs.(sensitivity),
            xlabel="Parameters",
            ylabel="|Sensitivity|",
            title=title,
            color=colors,
            alpha=0.7,
            legend=false,
            yaxis=:log,
            rotation=45,
            size=(800, 500))

    for (i, (name, sens)) in enumerate(zip(result.param_names, sensitivity))
        opt_val = result.optimized_params[i]
        annotate!(p, i, maximum(abs.(sensitivity)) * 0.9,
                  text(@sprintf("%.2e\n(opt=%.2f)", sens, opt_val), 8, :center))
    end

    return p
end

function animate_inundation(
    sol::ShallowWaterSolution,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    output_path::String;
    t_start::Float64=0.0,
    t_end::Float64=100.0,
    n_frames::Int=50,
    fps::Int=10,
    cmap::Symbol=:YlOrRd,
    depth_threshold::Float64=0.5
)
    times = range(t_start, t_end, length=n_frames)

    anim = @animate for t in times
        state = sol(t)
        inundation = compute_inundation_depth(state, domain)
        risk_map = compute_inundation_risk_map(sites, inundation, domain.x, domain.y;
                                               depth_threshold=depth_threshold)

        p1 = heatmap(domain.x[:, 1], domain.y[1, :], inundation',
                     xlabel="X (m)", ylabel="Y (m)",
                     title=@sprintf("Inundation Depth (t=%.1f)", t),
                     color=:blues,
                     colorbar_title="Depth (m)",
                     aspect_ratio=:equal,
                     clims=(0, max(1, maximum(inundation))))

        for site in sites
            scatter!(p1, [site.x], [site.y],
                    markersize=max(6, site.radius / 8),
                    marker=:circle,
                    markeralpha=0.8,
                    markercolor=:red,
                    label="")
        end

        p2 = heatmap(domain.x[:, 1], domain.y[1, :], risk_map',
                     xlabel="X (m)", ylabel="Y (m)",
                     title="Inundation Risk",
                     color=cmap,
                     colorbar_title="Risk",
                     aspect_ratio=:equal,
                     clims=(0, max(1, maximum(risk_map))))

        for site in sites
            site_risk = compute_site_inundation_risk(site, inundation, domain.x, domain.y;
                                                     depth_threshold=depth_threshold)
            color = site_risk > 0.5 ? :red : (site_risk > 0.2 ? :orange : :green)
            scatter!(p2, [site.x], [site.y],
                    markersize=max(6, site.radius / 8),
                    marker=:circle,
                    markeralpha=0.8,
                    markercolor=color,
                    markerstrokecolor=:black,
                    markerstrokewidth=2,
                    label="")
        end

        l = @layout [a b]
        plot(p1, p2, layout=l, size=(1200, 500))
    end

    gif(anim, output_path, fps=fps)
    return anim
end

function plot_site_locations(
    sites::Vector{ArchaeologicalSite},
    domain::ShallowWaterDomain;
    title::String="Archaeological Site Locations",
    show_time_periods::Bool=true
)
    p = scatter(title=title,
                xlabel="X (m)", ylabel="Y (m)",
                aspect_ratio=:equal,
                xlims=(minimum(domain.x), maximum(domain.x)),
                ylims=(minimum(domain.y), maximum(domain.y)),
                size=(800, 600))

    cultures = unique([s.culture for s in sites])
    colors = distinguishable_colors(length(cultures), [RGB(1,1,1), RGB(0,0,0)])
    culture_colors = Dict(zip(cultures, colors))

    for site in sites
        color = culture_colors[site.culture]

        theta = range(0, 2π, length=50)
        x_circle = site.x .+ site.radius * cos.(theta)
        y_circle = site.y .+ site.radius * sin.(theta)
        plot!(p, x_circle, y_circle,
              fill=true,
              fillalpha=0.2,
              color=color,
              linealpha=0.5,
              label="")

        scatter!(p, [site.x], [site.y],
                markersize=8,
                marker=:circle,
                markercolor=color,
                markeralpha=0.8,
                markerstrokewidth=2,
                label=@sprintf("%s (%s)\n%.0f-%.0f BP", site.id, site.culture, site.start_age, site.end_age))
    end

    if show_time_periods
        p_time = plot(title="Chronology",
                      yticks=(1:length(sites), [s.id for s in sites]),
                      xlabel="Time (years BP)",
                      ylabel="Sites",
                      size=(800, 300),
                      legend=false)

        for (i, site) in enumerate(sites)
            color = culture_colors[site.culture]
            plot!(p_time, [site.start_age, site.end_age], [i, i],
                  linewidth=8,
                  color=color,
                  alpha=0.7)
        end
    end

    l = @layout [a{0.6h}; b{0.4h}]
    p_final = plot(p, p_time, layout=l, size=(1000, 900))

    return p_final
end

end
