includet("load_demo_data.jl")

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic_Data/raw_000.data", use_float32=true, use_nom_kz=true);

using Optim
using ReadWriteCFL

nx = 256
ny = 256
nz = 32

#######################################################################################################################


echo_times = time_since_last_rf[1,:,1,1]

print("echo times")
println(echo_times)

x = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/x_2d")

# Function to model the exponential decay
function exp_decay(t, S0, T2_star)
    return S0 * exp(-t / T2_star)
end

# Function to fit the T2* value for a given voxel (x, y, z)
function fit_t2_star(voxel_data, echo_times)
    # Define the objective function (sum of squared residuals)
    obj_func(S) = sum((voxel_data .- exp_decay.(echo_times, S[1], S[2])).^2)
    
    # Initial guess: S0 = max signal, T2* = 50ms
    init_guess = [maximum(voxel_data), 50.0]
    
    # Minimize the objective function using Levenberg-Marquardt
    result = optimize(obj_func, init_guess, NelderMead())
    
    # The fitted parameters
    return result.minimizer[2]  # T2* value
end

# Initialize an array to store T2* values
t2_star_map = Array{Float32}(undef, nx, ny, nz);

# Loop over each voxel and fit the T2* value
for ix in 1:nx
    for iy in 1:ny
        for iz in 1:nz
            voxel_data = abs.(x[ix, iy, iz, :])  # Assuming data is complex, use magnitude
            t2_star_map[ix, iy, iz] = fit_t2_star(voxel_data, echo_times)
        end
    end
end

ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/t2star_2d", ComplexF32.(t2_star_map))