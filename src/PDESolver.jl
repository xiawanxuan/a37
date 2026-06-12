module PDESolver

using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Statistics

export
    ShallowWaterParams,
    ShallowWaterDomain,
    ShallowWaterState,
    ShallowWaterSolution,
    solve_shallow_water,
    compute_inundation_depth,
    compute_flow_velocity

struct ShallowWaterParams
    g::Float64
    n::Float64
    ν::Float64
    h₀::Float64
    dt_max::Float64
end

function ShallowWaterParams(;
    g::Float64=9.81,
    n::Float64=0.03,
    ν::Float64=1.0,
    h₀::Float64=1e-3,
    dt_max::Float64=0.1
)
    return ShallowWaterParams(g, n, ν, h₀, dt_max)
end

struct ShallowWaterDomain
    Nx::Int
    Ny::Int
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::Matrix{Float64}
    y::Matrix{Float64}
    zb::Matrix{Float64}
end

function ShallowWaterDomain(
    Nx::Int, Ny::Int, Lx::Float64, Ly::Float64,
    zb::Matrix{Float64}=zeros(Nx, Ny)
)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = [i * dx for i in 0:Nx-1, j in 0:Ny-1]
    y = [j * dy for i in 0:Nx-1, j in 0:Ny-1]
    return ShallowWaterDomain(Nx, Ny, Lx, Ly, dx, dy, x, y, zb)
end

struct ShallowWaterState
    h::Matrix{Float64}
    u::Matrix{Float64}
    v::Matrix{Float64}
end

function ShallowWaterState(Nx::Int, Ny::Int; h_init::Float64=0.0)
    return ShallowWaterState(
        fill(h_init, Nx, Ny),
        zeros(Nx, Ny),
        zeros(Nx, Ny)
    )
end

function shallow_water_rhs!(dudt::Vector{Float64}, u::Vector{Float64},
                           params::Tuple{ShallowWaterParams, ShallowWaterDomain}, t::Float64)
    swp, domain = params
    Nx, Ny = domain.Nx, domain.Ny
    dx, dy = domain.dx, domain.dy
    g, n, ν, h₀ = swp.g, swp.n, swp.ν, swp.h₀

    h = reshape(u[1:Nx*Ny], Nx, Ny)
    u_vel = reshape(u[Nx*Ny+1:2*Nx*Ny], Nx, Ny)
    v_vel = reshape(u[2*Nx*Ny+1:3*Nx*Ny], Nx, Ny)

    dhdt = reshape(dudt[1:Nx*Ny], Nx, Ny)
    dudt_vel = reshape(dudt[Nx*Ny+1:2*Nx*Ny], Nx, Ny)
    dvdt_vel = reshape(dudt[2*Nx*Ny+1:3*Nx*Ny], Nx, Ny)

    zb = domain.zb
    h_total = max.(h, h₀)
    h_safe = max.(h, h₀)

    qx = h_safe .* u_vel
    qy = h_safe .* v_vel

    @inline function gradient_x(f::Matrix{Float64}, dx::Float64)
        grad = zeros(Nx, Ny)
        for j in 1:Ny
            for i in 2:Nx-1
                grad[i, j] = (f[i+1, j] - f[i-1, j]) / (2 * dx)
            end
            grad[1, j] = (f[2, j] - f[1, j]) / dx
            grad[Nx, j] = (f[Nx, j] - f[Nx-1, j]) / dx
        end
        return grad
    end

    @inline function gradient_y(f::Matrix{Float64}, dy::Float64)
        grad = zeros(Nx, Ny)
        for i in 1:Nx
            for j in 2:Ny-1
                grad[i, j] = (f[i, j+1] - f[i, j-1]) / (2 * dy)
            end
            grad[i, 1] = (f[i, 2] - f[i, 1]) / dy
            grad[i, Ny] = (f[i, Ny] - f[i, Ny-1]) / dy
        end
        return grad
    end

    @inline function laplacian(f::Matrix{Float64}, dx::Float64, dy::Float64)
        lap = zeros(Nx, Ny)
        for j in 2:Ny-1
            for i in 2:Nx-1
                lap[i, j] = (f[i+1, j] - 2*f[i, j] + f[i-1, j]) / dx^2 +
                           (f[i, j+1] - 2*f[i, j] + f[i, j-1]) / dy^2
            end
        end
        return lap
    end

    dqx_dx = gradient_x(qx, dx)
    dqy_dy = gradient_y(qy, dy)

    dhdt .= -(dqx_dx + dqy_dy)

    d(qx²/h_safe)_dx = gradient_x(qx.^2 ./ h_safe, dx)
    d(qx*qy/h_safe)_dy = gradient_y(qx .* qy ./ h_safe, dy)
    d(gh²/2 + g*zb*h_safe)_dx = gradient_x(g .* (h_safe.^2 / 2 + zb .* h_safe), dx)

    mag_vel = sqrt.(u_vel.^2 + v_vel.^2)
    stress_x = g * n^2 .* mag_vel .* u_vel ./ max.(h_safe, h₀).^(1/3)

    dudt_vel .= -(d(qx²/h_safe)_dx + d(qx*qy/h_safe)_dy + d(gh²/2 + g*zb*h_safe)_dx) ./ h_safe - stress_x + ν .* laplacian(u_vel, dx, dy)

    d(qx*qy/h_safe)_dx = gradient_x(qx .* qy ./ h_safe, dx)
    d(qy²/h_safe)_dy = gradient_y(qy.^2 ./ h_safe, dy)
    d(gh²/2 + g*zb*h_safe)_dy = gradient_y(g .* (h_safe.^2 / 2 + zb .* h_safe), dy)

    stress_y = g * n^2 .* mag_vel .* v_vel ./ max.(h_safe, h₀).^(1/3)

    dvdt_vel .= -(d(qx*qy/h_safe)_dx + d(qy²/h_safe)_dy + d(gh²/2 + g*zb*h_safe)_dy) ./ h_safe - stress_y + ν .* laplacian(v_vel, dx, dy)

    apply_boundary_conditions!(dhdt, dudt_vel, dvdt_vel, h, u_vel, v_vel, Nx, Ny, h₀)

    return nothing
end

function apply_boundary_conditions!(dhdt::Matrix{Float64}, dudt::Matrix{Float64}, dvdt::Matrix{Float64},
                                    h::Matrix{Float64}, u::Matrix{Float64}, v::Matrix{Float64},
                                    Nx::Int, Ny::Int, h₀::Float64)
    for j in 1:Ny
        dhdt[1, j] = 0.0
        dudt[1, j] = 0.0
        dvdt[1, j] = 0.0
        dhdt[Nx, j] = 0.0
        dudt[Nx, j] = 0.0
        dvdt[Nx, j] = 0.0
    end
    for i in 1:Nx
        dhdt[i, 1] = 0.0
        dudt[i, 1] = 0.0
        dvdt[i, 1] = 0.0
        dhdt[i, Ny] = 0.0
        dudt[i, Ny] = 0.0
        dvdt[i, Ny] = 0.0
    end
    return nothing
end

function apply_initial_conditions!(state::ShallowWaterState, domain::ShallowWaterDomain,
                                   params::ShallowWaterParams, inflow_rate::Float64=1.0)
    Nx, Ny = domain.Nx, domain.Ny
    h₀ = params.h₀

    channel_center_y = domain.Ly / 2
    channel_width = domain.Ly / 6

    for j in 1:Ny
        for i in 1:Nx
            y = domain.y[i, j]
            dist_to_channel = abs(y - channel_center_y)
            if dist_to_channel < channel_width
                state.h[i, j] = max(0.5, inflow_rate * (1 - dist_to_channel / channel_width))
                state.u[i, j] = inflow_rate * 0.5 * (1 - dist_to_channel / channel_width)
            else
                state.h[i, j] = h₀
                state.u[i, j] = 0.0
            end
            state.v[i, j] = 0.0
        end
    end

    return nothing
end

function solve_shallow_water(
    domain::ShallowWaterDomain,
    params::ShallowWaterParams,
    tspan::Tuple{Float64, Float64};
    state0::Union{ShallowWaterState, Nothing}=nothing,
    inflow_rate::Float64=1.0,
    saveat::Vector{Float64}=Float64[],
    alg=Tsit5()
)
    Nx, Ny = domain.Nx, domain.Ny

    if state0 === nothing
        state0 = ShallowWaterState(Nx, Ny)
        apply_initial_conditions!(state0, domain, params, inflow_rate)
    end

    u0 = vcat(vec(state0.h), vec(state0.u), vec(state0.v))
    p = (params, domain)

    prob = ODEProblem(shallow_water_rhs!, u0, tspan, p)

    if isempty(saveat)
        sol = solve(prob, alg; adaptive=true, dt=1e-3, maxiters=10^7,
                   abstol=1e-3, reltol=1e-3)
    else
        sol = solve(prob, alg; adaptive=true, dt=1e-3, maxiters=10^7,
                   abstol=1e-3, reltol=1e-3, saveat=saveat)
    end

    return ShallowWaterSolution(sol, domain, params)
end

struct ShallowWaterSolution
    sol::ODESolution
    domain::ShallowWaterDomain
    params::ShallowWaterParams
end

function (sol::ShallowWaterSolution)(t::Float64)
    u = sol.sol(t)
    Nx, Ny = sol.domain.Nx, sol.domain.Ny
    return ShallowWaterState(
        reshape(u[1:Nx*Ny], Nx, Ny),
        reshape(u[Nx*Ny+1:2*Nx*Ny], Nx, Ny),
        reshape(u[2*Nx*Ny+1:3*Nx*Ny], Nx, Ny)
    )
end

function compute_inundation_depth(state::ShallowWaterState, domain::ShallowWaterDomain)
    return max.(state.h .- domain.zb, 0.0)
end

function compute_flow_velocity(state::ShallowWaterState)
    return sqrt.(state.u.^2 + state.v.^2)
end

end
