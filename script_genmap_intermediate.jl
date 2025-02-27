includet("load_demo_data.jl")

using LsqFit
using ReadWriteCFL
using Plots

nx = 256
ny = 256
nz = 32

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic_Data/raw_000.data", use_float32=true, use_nom_kz=true)

echo_times = time_since_last_rf[1,:,1,1]

println("Echo times: ", echo_times)

x = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/x_2d")

# Function to model the exponential decay
function exp_decay(t, S)
    S0, T2_star = S
    return S0 .* exp.(-t ./ T2_star)  # Ensure element-wise operations
end

# Function to fit T2* value for a given voxel (x, y, z)
function fit_t2_star(voxel_data, echo_times)
    # if maximum(voxel_data) < 1e-5  # Avoid fitting on very low signal
    #     return NaN
    # end

    # Initial guess: S0 = max signal, T2* = 50ms
    init_guess = [voxel_data[1], 50.0]

    # Curve fitting
    fit = curve_fit(exp_decay, echo_times, voxel_data, init_guess,lower=[0.0, 0.0], upper=[voxel_data[1], 500.0])

    return fit.param  # Extract fitted T2* value
end

# Initialize an array to store T2* values
t2_star_map = Array{Float32}(undef, nx, ny, nz)
s0_map = Array{Float32}(undef, nx, ny, nz)

# Loop over each voxel and fit the T2* value
for ix in 1:nx
    for iy in 1:ny
        for iz in 1:nz
            voxel_data = abs.(x[ix, iy, iz, :])  # Ensure magnitude is used
            s0_map[ix,iy,iz], t2_star_map[ix, iy, iz] = fit_t2_star(voxel_data, echo_times)
        end
    end
end

# Save T2* map
ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/t2star_2d", ComplexF32.(t2_star_map))

function visualize_slices(t2_star_map)
    anim = @animate for i in 1:size(t2_star_map,3)
        heatmap(t2_star_map[:,:,i], color=:viridis, title="Slice $i")
    end
    gif(anim,"t2_star-map.gif",fps=5)
end

visualize_slices(t2_star_map)
