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
    h_dry::Float64
end

function ShallowWaterParams(;
    g::Float64=9.81,
    n::Float64=0.03,
    ν::Float64=1.0,
    h₀::Float64=1e-3,
    dt_max::Float64=0.1,
    h_dry::Float64=0.01
)
    return ShallowWaterParams(g, n, ν, h₀, dt_max, h_dry)
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
    g, n, ν, h₀, h_dry = swp.g, swp.n, swp.ν, swp.h₀, swp.h_dry

    h = reshape(u[1:Nx*Ny], Nx, Ny)
    u_vel = reshape(u[Nx*Ny+1:2*Nx*Ny], Nx, Ny)
    v_vel = reshape(u[2*Nx*Ny+1:3*Nx*Ny], Nx, Ny)

    dhdt = reshape(dudt[1:Nx*Ny], Nx, Ny)
    dudt_vel = reshape(dudt[Nx*Ny+1:2*Nx*Ny], Nx, Ny)
    dvdt_vel = reshape(dudt[2*Nx*Ny+1:3*Nx*Ny], Nx, Ny)

    zb = domain.zb

    for idx in eachindex(h)
        if h[idx] < h₀
            h[idx] = h₀
        end
        if isnan(h[idx]) || isnan(u_vel[idx]) || isnan(v_vel[idx])
            h[idx] = h₀
            u_vel[idx] = 0.0
            v_vel[idx] = 0.0
        end
    end

    wet = h .>= h_dry

    h_safe = max.(h, h_dry)

    u_vel_wet = u_vel .* wet
    v_vel_wet = v_vel .* wet

    qx = h_safe .* u_vel_wet
    qy = h_safe .* v_vel_wet

    @inline function minmod(a::Float64, b::Float64)
        if a * b <= 0.0
            return 0.0
        elseif abs(a) < abs(b)
            return a
        else
            return b
        end
    end

    @inline function gradient_x_limited(f::Matrix{Float64}, dx::Float64)
        grad = zeros(Nx, Ny)
        for j in 1:Ny
            for i in 2:Nx-1
                forward = (f[i+1, j] - f[i, j]) / dx
                backward = (f[i, j] - f[i-1, j]) / dx
                grad[i, j] = minmod(forward, backward)
            end
            grad[1, j] = (f[2, j] - f[1, j]) / dx
            grad[Nx, j] = (f[Nx, j] - f[Nx-1, j]) / dx
        end
        return grad
    end

    @inline function gradient_y_limited(f::Matrix{Float64}, dy::Float64)
        grad = zeros(Nx, Ny)
        for i in 1:Nx
            for j in 2:Ny-1
                forward = (f[i, j+1] - f[i, j]) / dy
                backward = (f[i, j] - f[i, j-1]) / dy
                grad[i, j] = minmod(forward, backward)
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

    dqx_dx = gradient_x_limited(qx, dx)
    dqy_dy = gradient_y_limited(qy, dy)

    dhdt .= -(dqx_dx + dqy_dy)

    pressure_x = gradient_x_limited(g .* (h_safe.^2 / 2 .+ zb .* h_safe), dx)
    advect_xx = gradient_x_limited(qx.^2 ./ h_safe, dx)
    advect_xy = gradient_y_limited(qx .* qy ./ h_safe, dy)

    mag_vel = sqrt.(u_vel_wet.^2 + v_vel_wet.^2)
    friction_x = g * n^2 .* mag_vel .* u_vel_wet ./ (h_safe.^(4/3))
    friction_x .*= wet

    lap_u = laplacian(u_vel_wet, dx, dy)

    dudt_vel .= wet .* (-(advect_xx + advect_xy + pressure_x) ./ h_safe - friction_x + ν .* lap_u)

    pressure_y = gradient_y_limited(g .* (h_safe.^2 / 2 .+ zb .* h_safe), dy)
    advect_yx = gradient_x_limited(qx .* qy ./ h_safe, dx)
    advect_yy = gradient_y_limited(qy.^2 ./ h_safe, dy)

    friction_y = g * n^2 .* mag_vel .* v_vel_wet ./ (h_safe.^(4/3))
    friction_y .*= wet

    lap_v = laplacian(v_vel_wet, dx, dy)

    dvdt_vel .= wet .* (-(advect_yx + advect_yy + pressure_y) ./ h_safe - friction_y + ν .* lap_v)

    apply_boundary_conditions!(dhdt, dudt_vel, dvdt_vel, h, u_vel, v_vel, Nx, Ny, h₀)

    for idx in eachindex(dudt)
        if isnan(dudt[idx])
            dudt[idx] = 0.0
        end
    end

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
