includet("../load_demo_data.jl")

using LsqFit
using ReadWriteCFL
using Plots
using Statistics

nx = 256
ny = 256
nz = 32

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true)

combine_coils = true
use_dcf = true

γ = 2 * π * 42.576e6

x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);
Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
s0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
ϕs0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

echo_times = time_since_last_rf[268,:,1,1]

# println("Echo times: ", echo_times)

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""

x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$comb$dcf")
Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$comb$dcf"))
ϕs0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/init_phase$comb$dcf"))

println("Data Loaded")

mag = mean(abs.(x), dims=4)   # → size (nx,ny,nz)
mask = dropdims(mag .> 1e-7, dims=4)   # boolean mask of in-object voxels

# Function to model the exponential decay
function forward_model(t,p, ϕs0, Δb0)
    s0_mag, R2_star = p
    S0 = ComplexF64(s0_mag * cis(ϕs0))
    signal = S0 .* exp.(im * γ * Δb0 * t - (t .* R2_star))
    return vcat(real.(signal), imag.(signal))
end

function fit_s0_r2(voxel_data, echo_times, ϕs0_p, Δb0)

    ydata = vcat(real(voxel_data), imag(voxel_data))

    init_guess = [abs(voxel_data[1]), 1 / 50.0]
    lower = [0.0, 1.0 / 500.0]
    upper = [Inf, 1.0]

    model = (x,p) -> forward_model(x,p, ϕs0_p, Δb0)
    fit = curve_fit(model, echo_times, ydata, init_guess; lower=lower, upper=upper)

    S0_magnitude, R2_star = fit.param

    S0 = ComplexF64(S0_magnitude * cis(ϕs0_p))
    
    return S0, R2_star
end

# Initialize an array to store R2* values
r2 =  combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
# s0_magnitude =  combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

# Loop over each voxel and fit the R2* value
if combine_coils
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        if mask[ix, iy, iz]
            s0[ix, iy, iz], r2[ix, iy, iz] = fit_s0_r2(@view(x[ix, iy, iz, :]), echo_times, ϕs0[ix,iy,iz], Δb0[ix, iy,iz])
        else
            s0[ix,iy,iz] = 0+0im
            r2[ix,iy,iz] = NaN
        end
    end
else
    for ichan in 1:config["nchan"]
        @info "channel = $ichan"
        for ix in 1:nx, iy in 1:ny, iz in 1:nz
            if mask[ix, iy, iz]
                s0[ix, iy, iz, ichan], r2[ix, iy, iz, ichan] = fit_s0_r2(@view(x[ix, iy, iz, ichan, :]), echo_times, ϕs0[ix,iy,iz,ichan], Δb0[ix, iy,iz,ichan])
            else
                s0[ix,iy,iz, ichan] = 0+0im
                r2[ix,iy,iz, ichan] = NaN
            end
        end
    end
end

function visualize_slices(r2)
    anim = @animate for (i, slice) in enumerate(eachslice(1 ./ r2; dims=3))
        heatmap(slice, color=:viridis, title="Slice $i")
    end
    gif(anim, "t2_star-map.gif", fps=5)
end

# s0 .= ComplexF64.(s0_magnitude .* cis.(ϕs0))

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/t2$comb$dcf", ComplexF32.(1.0 ./ r2))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$comb$dcf", ComplexF32.(s0))

# visualize_slices( 1 ./ r2[:,:,:,1])
