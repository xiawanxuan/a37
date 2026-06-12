module SiteCoupling

using LinearAlgebra
using Statistics
using Random
using Interpolations

export
    ArchaeologicalSite,
    SiteChronology,
    create_topography_with_sites,
    apply_site_boundary_conditions!,
    compute_site_inundation_risk,
    compute_inundation_risk_map,
    update_topography_for_time_period!,
    generate_synthetic_sites,
    get_active_sites

struct ArchaeologicalSite
    id::String
    x::Float64
    y::Float64
    elevation::Float64
    radius::Float64
    start_age::Float64
    end_age::Float64
    culture::String
end

function ArchaeologicalSite(
    id::String, x::Float64, y::Float64, elevation::Float64, radius::Float64,
    start_age::Float64, end_age::Float64; culture::String="Unknown"
)
    return ArchaeologicalSite(id, x, y, elevation, radius, start_age, end_age, culture)
end

struct SiteChronology
    sites::Vector{ArchaeologicalSite}
    time_periods::Vector{Tuple{Float64, Float64}}
    period_names::Vector{String}
end

function SiteChronology(sites::Vector{ArchaeologicalSite})
    ages = vcat([[s.start_age, s.end_age] for s in sites]...)
    min_age = minimum(ages)
    max_age = maximum(ages)

    n_periods = 5
    period_edges = range(min_age, max_age, length=n_periods+1)
    time_periods = [(period_edges[i], period_edges[i+1]) for i in 1:n_periods]
    period_names = ["Period $i" for i in 1:n_periods]

    return SiteChronology(sites, time_periods, period_names)
end

function SiteChronology(sites::Vector{ArchaeologicalSite},
                        time_periods::Vector{Tuple{Float64, Float64}},
                        period_names::Vector{String})
    return SiteChronology(sites, time_periods, period_names)
end

function gaussian_elevation(
    x::Float64, y::Float64, site_x::Float64, site_y::Float64,
    height::Float64, radius::Float64
)
    dist_sq = (x - site_x)^2 + (y - site_y)^2
    return height * exp(-dist_sq / (2 * radius^2))
end

function create_topography_with_sites(
    domain_x::Matrix{Float64},
    domain_y::Matrix{Float64},
    sites::Vector{ArchaeologicalSite};
    base_topography::Union{Matrix{Float64}, Nothing}=nothing,
    base_slope::Float64=0.001,
    river_channel_depth::Float64=2.0,
    river_width::Float64=100.0,
    river_center_y::Union{Float64, Nothing}=nothing
)
    Nx, Ny = size(domain_x)
    Lx = maximum(domain_x) - minimum(domain_x)
    Ly = maximum(domain_y) - minimum(domain_y)

    if river_center_y === nothing
        river_center_y = Ly / 2
    end

    if base_topography === nothing
        zb = zeros(Nx, Ny)
        for j in 1:Ny
            for i in 1:Nx
                x = domain_x[i, j]
                y = domain_y[i, j]

                zb[i, j] = base_slope * (Lx - x)

                dist_to_river = abs(y - river_center_y)
                river_profile = river_channel_depth * exp(-dist_to_river^2 / (2 * river_width^2))
                zb[i, j] -= river_profile

                noise = 0.1 * randn()
                zb[i, j] += noise
            end
        end
    else
        zb = copy(base_topography)
    end

    for site in sites
        site_elevation = site.elevation
        for j in 1:Ny
            for i in 1:Nx
                x = domain_x[i, j]
                y = domain_y[i, j]
                zb[i, j] += gaussian_elevation(x, y, site.x, site.y, site_elevation, site.radius)
            end
        end
    end

    return zb
end

function update_topography_for_time_period!(
    zb::Matrix{Float64},
    domain_x::Matrix{Float64},
    domain_y::Matrix{Float64},
    sites::Vector{ArchaeologicalSite},
    current_time::Float64
)
    Nx, Ny = size(domain_x)

    for site in sites
        if site.start_age <= current_time <= site.end_age
            for j in 1:Ny
                for i in 1:Nx
                    x = domain_x[i, j]
                    y = domain_y[i, j]
                    elevation = gaussian_elevation(x, y, site.x, site.y, site.elevation, site.radius)
                    zb[i, j] += elevation
                end
            end
        end
    end

    return zb
end

function apply_site_boundary_conditions!(
    state,
    domain,
    sites::Vector{ArchaeologicalSite},
    current_time::Float64,
    params
)
    h_min = params.h₀
    Nx, Ny = domain.Nx, domain.Ny

    for site in sites
        if site.start_age <= current_time <= site.end_age
            for j in 1:Ny
                for i in 1:Nx
                    x = domain.x[i, j]
                    y = domain.y[i, j]
                    dist_sq = (x - site.x)^2 + (y - site.y)^2
                    if dist_sq < site.radius^2
                        elevation_factor = gaussian_elevation(x, y, site.x, site.y, 1.0, site.radius)
                        state.h[i, j] = max(state.h[i, j], h_min * (1 + 0.5 * elevation_factor))
                        state.u[i, j] *= (1 - 0.3 * elevation_factor)
                        state.v[i, j] *= (1 - 0.3 * elevation_factor)
                    end
                end
            end
        end
    end

    return state
end

function compute_site_inundation_risk(
    site::ArchaeologicalSite,
    inundation_depth::Matrix{Float64},
    domain_x::Matrix{Float64},
    domain_y::Matrix{Float64};
    depth_threshold::Float64=0.5
)
    Nx, Ny = size(inundation_depth)
    risk = 0.0
    total_weight = 0.0

    for j in 1:Ny
        for i in 1:Nx
            x = domain_x[i, j]
            y = domain_y[i, j]
            weight = gaussian_elevation(x, y, site.x, site.y, 1.0, site.radius)
            depth = inundation_depth[i, j]

            if depth > depth_threshold
                risk += weight * (depth / depth_threshold)
            end
            total_weight += weight
        end
    end

    return total_weight > 0 ? risk / total_weight : 0.0
end

function compute_inundation_risk_map(
    sites::Vector{ArchaeologicalSite},
    inundation_depth::Matrix{Float64},
    domain_x::Matrix{Float64},
    domain_y::Matrix{Float64};
    depth_threshold::Float64=0.5
)
    Nx, Ny = size(inundation_depth)
    risk_map = zeros(Nx, Ny)

    for site in sites
        site_risk = compute_site_inundation_risk(site, inundation_depth, domain_x, domain_y;
                                                 depth_threshold=depth_threshold)
        for j in 1:Ny
            for i in 1:Nx
                x = domain_x[i, j]
                y = domain_y[i, j]
                weight = gaussian_elevation(x, y, site.x, site.y, 1.0, site.radius)
                risk_map[i, j] += site_risk * weight
            end
        end
    end

    return risk_map
end

function generate_synthetic_sites(
    Lx::Float64, Ly::Float64,
    n_sites::Int=10;
    min_elevation::Float64=1.0,
    max_elevation::Float64=5.0,
    min_radius::Float64=20.0,
    max_radius::Float64=80.0,
    min_age::Float64=1000.0,
    max_age::Float64=10000.0,
    seed::Union{Int, Nothing}=nothing
)
    if seed !== nothing
        Random.seed!(seed)
    end

    sites = ArchaeologicalSite[]
    river_center_y = Ly / 2

    for i in 1:n_sites
        id = "Site_$i"
        x = rand() * Lx
        y = rand() * Ly

        dist_to_river = abs(y - river_center_y)
        elevation_pref = exp(-dist_to_river^2 / (2 * (Ly/4)^2))
        elevation = min_elevation + (max_elevation - min_elevation) * elevation_pref * rand()

        radius = min_radius + (max_radius - min_radius) * rand()

        start_age = min_age + (max_age - min_age) * rand()
        duration = 500 + 2000 * rand()
        end_age = min(max_age, start_age + duration)

        cultures = ["Yangshao", "Longshan", "Erlitou", "Shang", "Zhou"]
        culture = cultures[rand(1:length(cultures))]

        push!(sites, ArchaeologicalSite(id, x, y, elevation, radius, start_age, end_age, culture))
    end

    return sites
end

function get_active_sites(sites::Vector{ArchaeologicalSite}, current_time::Float64)
    return [s for s in sites if s.start_age <= current_time <= s.end_age]
end

end
