module ParameterInversion

using Optim
using LineSearches
using NLSolversBase
using LinearAlgebra
using Statistics
using Printf

import ..PDESolver:
    ShallowWaterParams,
    ShallowWaterDomain,
    ShallowWaterState,
    solve_shallow_water,
    compute_inundation_depth,
    compute_flow_velocity,
    ShallowWaterSolution

import ..SiteCoupling:
    ArchaeologicalSite,
    SiteChronology,
    create_topography_with_sites,
    compute_site_inundation_risk,
    get_active_sites

export
    PaleoriverParams,
    InversionParams,
    InversionResult,
    inversion_objective,
    invert_paleoriver_params,
    generate_observations,
    compute_param_sensitivity,
    param_vec_to_struct,
    struct_to_param_vec,
    simulate_inundation,
    compute_inundation_observations

struct PaleoriverParams
    inflow_rate::Float64
    river_center_y::Float64
    river_width::Float64
    river_depth::Float64
    roughness::Float64
    sediment_flux::Float64

    function PaleoriverParams(
        inflow_rate::Float64,
        river_center_y::Float64,
        river_width::Float64,
        river_depth::Float64,
        roughness::Float64,
        sediment_flux::Float64=0.0
    )
        return new(inflow_rate, river_center_y, river_width, river_depth, roughness, sediment_flux)
    end
end

function PaleoriverParams(;
    inflow_rate::Float64=1.0,
    river_center_y::Float64=500.0,
    river_width::Float64=100.0,
    river_depth::Float64=2.0,
    roughness::Float64=0.03,
    sediment_flux::Float64=0.0
)
    return PaleoriverParams(inflow_rate, river_center_y, river_width, river_depth, roughness, sediment_flux)
end

struct InversionParams
    param_names::Vector{String}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    initial_params::Vector{Float64}
    max_iterations::Int
    tolerance::Float64
    algorithm::Symbol

    function InversionParams(
        param_names::Vector{String},
        lower_bounds::Vector{Float64},
        upper_bounds::Vector{Float64},
        initial_params::Vector{Float64};
        max_iterations::Int=100,
        tolerance::Float64=1e-6,
        algorithm::Symbol=:LBFGS
    )
        @assert length(param_names) == length(lower_bounds) == length(upper_bounds) == length(initial_params)
        return new(param_names, lower_bounds, upper_bounds, initial_params,
                   max_iterations, tolerance, algorithm)
    end
end

struct InversionResult
    optimized_params::Vector{Float64}
    param_names::Vector{String}
    minimum_objective::Float64
    iterations::Int
    converged::Bool
    trace::Vector{Float64}
    param_history::Matrix{Float64}
end

function param_vec_to_struct(params_vec::Vector{Float64}, param_names::Vector{String})
    param_dict = Dict(zip(param_names, params_vec))

    return PaleoriverParams(
        inflow_rate=get(param_dict, "inflow_rate", 1.0),
        river_center_y=get(param_dict, "river_center_y", 500.0),
        river_width=get(param_dict, "river_width", 100.0),
        river_depth=get(param_dict, "river_depth", 2.0),
        roughness=get(param_dict, "roughness", 0.03),
        sediment_flux=get(param_dict, "sediment_flux", 0.0)
    )
end

function struct_to_param_vec(params::PaleoriverParams, param_names::Vector{String})
    param_dict = Dict(
        "inflow_rate" => params.inflow_rate,
        "river_center_y" => params.river_center_y,
        "river_width" => params.river_width,
        "river_depth" => params.river_depth,
        "roughness" => params.roughness,
        "sediment_flux" => params.sediment_flux
    )
    return [param_dict[name] for name in param_names]
end

function simulate_inundation(
    paleo_params::PaleoriverParams,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    tspan::Tuple{Float64, Float64},
    eval_times::Vector{Float64}
)
    sw_params = ShallowWaterParams(
        g=9.81,
        n=paleo_params.roughness,
        ν=1.0,
        h₀=1e-3
    )

    zb = create_topography_with_sites(
        domain.x, domain.y, sites;
        base_slope=0.001,
        river_channel_depth=paleo_params.river_depth,
        river_width=paleo_params.river_width,
        river_center_y=paleo_params.river_center_y
    )

    domain_with_sites = ShallowWaterDomain(domain.Nx, domain.Ny, domain.Lx, domain.Ly, zb)

    sol = solve_shallow_water(
        domain_with_sites, sw_params, tspan;
        inflow_rate=paleo_params.inflow_rate,
        saveat=eval_times
    )

    return sol, domain_with_sites
end

function compute_inundation_observations(
    sol::ShallowWaterSolution,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    eval_times::Vector{Float64}
)
    n_times = length(eval_times)
    n_sites = length(sites)

    observations = zeros(n_times, n_sites)

    for (t_idx, t) in enumerate(eval_times)
        state = sol(t)
        inundation = compute_inundation_depth(state, domain)

        for (s_idx, site) in enumerate(sites)
            if site.start_age <= t <= site.end_age
                risk = compute_site_inundation_risk(site, inundation, domain.x, domain.y)
                observations[t_idx, s_idx] = risk
            end
        end
    end

    return observations
end

function generate_observations(
    true_params::PaleoriverParams,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    tspan::Tuple{Float64, Float64},
    eval_times::Vector{Float64};
    noise_level::Float64=0.1
)
    sol, domain_with_sites = simulate_inundation(true_params, domain, sites, tspan, eval_times)
    true_obs = compute_inundation_observations(sol, domain_with_sites, sites, eval_times)

    noise = noise_level * std(true_obs) * randn(size(true_obs))
    noisy_obs = true_obs + noise
    noisy_obs = max.(0.0, noisy_obs)

    return noisy_obs, true_obs, sol, domain_with_sites
end

function inversion_objective(
    params_vec::Vector{Float64},
    inv_params::InversionParams,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    tspan::Tuple{Float64, Float64},
    eval_times::Vector{Float64},
    observations::Matrix{Float64};
    verbose::Bool=false
)
    lower = inv_params.lower_bounds
    upper = inv_params.upper_bounds

    barrier = 0.0
    for i in 1:length(params_vec)
        margin_lo = (params_vec[i] - lower[i]) / (upper[i] - lower[i])
        margin_hi = (upper[i] - params_vec[i]) / (upper[i] - lower[i])
        if margin_lo <= 0.0 || margin_hi <= 0.0
            return 1e10 + 1e6 * sum(abs.(params_vec - (lower + upper) / 2))
        end
        if margin_lo < 0.05
            barrier -= 1e-2 * log(margin_lo / 0.05)
        end
        if margin_hi < 0.05
            barrier -= 1e-2 * log(margin_hi / 0.05)
        end
    end

    param_scales = (upper - lower)
    normalized_residual = (params_vec - inv_params.initial_params) ./ param_scales

    paleo_params = param_vec_to_struct(params_vec, inv_params.param_names)

    try
        sol, domain_with_sites = simulate_inundation(paleo_params, domain, sites, tspan, eval_times)
        simulated = compute_inundation_observations(sol, domain_with_sites, sites, eval_times)

        residuals = (simulated - observations)
        obs_range = maximum(observations) - minimum(observations)
        if obs_range < 1e-8
            obs_range = 1.0
        end
        normalized_res = residuals / obs_range
        weights = (observations .> 0) .+ 0.1
        mse = sum(weights .* normalized_res.^2) / sum(weights)

        regularization = 1e-2 * sum(normalized_residual.^2)

        total_loss = mse + regularization + barrier

        if isnan(total_loss) || isinf(total_loss)
            return 1e10
        end

        if verbose
            @printf("Params: ")
            for (name, val) in zip(inv_params.param_names, params_vec)
                @printf("%s=%.3f ", name, val)
            end
            @printf("Loss: %.6f (mse=%.6f reg=%.6f bar=%.6f)\n", total_loss, mse, regularization, barrier)
        end

        return total_loss
    catch e
        if verbose
            println("Simulation failed with error: ", e)
        end
        return 1e10
    end
end

function invert_paleoriver_params(
    inv_params::InversionParams,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    tspan::Tuple{Float64, Float64},
    eval_times::Vector{Float64},
    observations::Matrix{Float64};
    verbose::Bool=true
)
    trace = Float64[]
    param_history = Vector{Float64}[]

    function callback(x)
        push!(trace, x.value)
        push!(param_history, copy(x.metadata["x"]))
        if verbose
            @printf("Iter %d: Loss = %.6f\n", length(trace), x.value)
        end
        return false
    end

    lower = inv_params.lower_bounds
    upper = inv_params.upper_bounds
    initial_x = inv_params.initial_params

    param_scales = (upper - lower)
    normalized_lower = zeros(length(lower))
    normalized_upper = ones(length(upper))
    normalized_init = (initial_x - lower) ./ param_scales

    function normalized_objective(x_norm::Vector{Float64})
        x_physical = lower + x_norm .* param_scales
        return inversion_objective(
            x_physical, inv_params, domain, sites, tspan, eval_times, observations;
            verbose=false
        )
    end

    inner_optimizer = LBFGS(
        m=10,
        linesearch=LineSearches.BackTracking(order=3),
        alphaguess=LineSearches.InitialStatic(alpha=0.01)
    )

    result = optimize(
        normalized_objective,
        normalized_lower,
        normalized_upper,
        normalized_init,
        Fminbox(inner_optimizer),
        Optim.Options(
            iterations=inv_params.max_iterations,
            f_tol=inv_params.tolerance,
            g_tol=1e-3,
            show_trace=verbose,
            extended_trace=true,
            callback=callback,
            time_limit=3600.0
        )
    )

    optimized_normalized = Optim.minimizer(result)
    optimized_params = lower + optimized_normalized .* param_scales
    min_obj = Optim.minimum(result)
    iterations = Optim.iterations(result)
    converged = Optim.converged(result)

    param_history_mat = isempty(param_history) ?
        zeros(length(initial_x), 1) :
        hcat(param_history...)

    return InversionResult(
        optimized_params,
        inv_params.param_names,
        min_obj,
        iterations,
        converged,
        trace,
        param_history_mat
    )
end

function compute_param_sensitivity(
    result::InversionResult,
    inv_params::InversionParams,
    domain::ShallowWaterDomain,
    sites::Vector{ArchaeologicalSite},
    tspan::Tuple{Float64, Float64},
    eval_times::Vector{Float64},
    observations::Matrix{Float64};
    epsilon::Float64=1e-3
)
    n_params = length(result.optimized_params)
    sensitivity = zeros(n_params)

    base_loss = inversion_objective(
        result.optimized_params, inv_params, domain, sites, tspan, eval_times, observations
    )

    for i in 1:n_params
        params_plus = copy(result.optimized_params)
        params_plus[i] += epsilon * abs(params_plus[i])
        loss_plus = inversion_objective(
            params_plus, inv_params, domain, sites, tspan, eval_times, observations
        )

        params_minus = copy(result.optimized_params)
        params_minus[i] -= epsilon * abs(params_minus[i])
        loss_minus = inversion_objective(
            params_minus, inv_params, domain, sites, tspan, eval_times, observations
        )

        sensitivity[i] = (loss_plus - loss_minus) / (2 * epsilon * abs(result.optimized_params[i]))
    end

    return sensitivity
end

end
