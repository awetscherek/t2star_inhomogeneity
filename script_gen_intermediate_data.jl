includet("load_demo_data.jl")

using LsqFit
using ReadWriteCFL
using Plots

nx = 256
ny = 256
nz = 32

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true)

combine_coils = true
use_dcf = true

γ = 2 * π * 42.576e6

x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);
Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
ϕs0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

echo_times = time_since_last_rf[268,:,1,1]

println("Echo times: ", echo_times)

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""

x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$comb$dcf")
Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$comb$dcf"))
ϕs0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/init_phase$comb$dcf"))

println("data loaded")

# Function to model the exponential decay
function forward_model(t,p, ϕs0, Δb0)
    s0_mag, T2_star = p
    if any(!isfinite, s0_mag) || any(!isfinite, T2_star) ||
        !isfinite(ϕs0) || !isfinite(Δb0)
         return (0.0, NaN)
    end
    S0 = ComplexF64(s0_mag * cis(ϕs0))
    signal = S0 .* exp.(im * γ * Δb0 * t - (t ./ T2_star))
    return vcat(real.(signal), imag.(signal))
end

function fit_t2_star(voxel_data, echo_times, s0_p, Δb0)

    ydata = vcat(real(voxel_data), imag(voxel_data))

    init_guess = [abs(voxel_data[1]), 50.0]
    lower = [0.0, 1.0]
    upper = [Inf, 1000.0]

    model = (x,p) -> forward_model(x,p, s0_p, Δb0)
    fit = curve_fit(model, echo_times, ydata, init_guess; lower=lower, upper=upper)

    S0_magnitude, T2_star = fit.param

    S0 = ComplexF64(S0_magnitude * cis(s0_p))
    return S0, T2_star
end

# Initialize an array to store T2* values
t2 =  combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
s0_magnitude =  combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

# Loop over each voxel and fit the T2* value
if combine_coils
    for ix in 1:nx
        for iy in 1:ny
            for iz in 1:nz
                s0_magnitude[ix, iy, iz], t2[ix, iy, iz] = fit_t2_star(@view(x[ix, iy, iz, :]), echo_times, ϕs0[ix,iy,iz], Δb0[ix, iy,iz])
            end
        end
    end
else
    for ichan in 1:config["nchan"]
        @info "channel = $ichan"
        for ix in 1:nx
            for iy in 1:ny
                for iz in 1:nz
                    s0_magnitude[ix, iy, iz, ichan], t2[ix, iy, iz, ichan] = fit_t2_star(@view(x[ix, iy, iz, ichan, :]), echo_times, ϕs0[ix,iy,iz,ichan], Δb0[ix, iy,iz,ichan])
                end
            end
        end
    end
end

function visualize_slices(t2)
    anim = @animate for (i, slice) in enumerate(eachslice(t2; dims=3))
        heatmap(slice, color=:viridis, title="Slice $i")
    end
    gif(anim, "t2_star-map.gif", fps=5)
end

s0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

s0 .= ComplexF64.(s0_magnitude .* cis.(ϕs0))

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/t2$comb$dcf", ComplexF32.(t2))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$comb$dcf", ComplexF32.(s0))

# visualize_slices(t2[:,:,:,1])
