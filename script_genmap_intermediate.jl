includet("load_demo_data.jl")

using LsqFit
using ReadWriteCFL
using Plots

nx = 256
ny = 256
nz = 32

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic_Data/raw_000.data", use_float32=true, use_nom_kz=true)

combine_coils = true

print("size since last rf")
println(size(time_since_last_rf))

echo_times = time_since_last_rf[268,:,1,1]

println("Echo times: ", echo_times)

if combine_coils
    #Shape (nx, ny, nz, necho)
    x = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/x_2d")
else
    #Shape (nx, ny, nz, nchan, necho)
    x = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/x_2d_no_combine_coils")
end

println("data loaded")

# Function to model the exponential decay
function exp_decay(t, p)
    S0_real, S0_imag, T2_star = p
    S0 = complex(S0_real, S0_imag)
    signal = S0 .* exp.(-t ./ T2_star)  # Return only real part for real-valued residuals
    return vcat(real(signal), imag(signal))
end

# Function to fit T2* value for a given voxel (x, y, z)
function fit_t2_star(voxel_data, echo_times)

    ydata = vcat(real(voxel_data), imag(voxel_data))

    # Use real and imag parts for initial guess
    init_guess = [0, 0, 50.0]
    lower = [-Inf, -Inf, 1.0]
    upper = [Inf, Inf, 1000.0]

    # Fit using only real part of the voxel data
    fit = curve_fit(exp_decay, echo_times, ydata, init_guess; lower=lower, upper=upper)

    S0_real, S0_imag, T2_star = fit.param
    S0 = complex(S0_real, S0_imag)
    return S0, T2_star
end

# Initialize an array to store T2* values
t2_star_map =  combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
s0_map =  combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

# abs_x = abs.(x)

# Loop over each voxel and fit the T2* value
if combine_coils
    for ix in 1:nx
        for iy in 1:ny
            for iz in 1:nz
                voxel_data = x[ix, iy, iz, :]
                s0_map[ix,iy,iz], t2_star_map[ix, iy, iz] = fit_t2_star(voxel_data, echo_times)
            end
        end
    end
else
    for ichan in 1:config["nchan"]
        @info "channel = $ichan"
        for ix in 1:nx
            for iy in 1:ny
                for iz in 1:nz
                    voxel_data = abs_x[ix, iy, iz, ichan, :]
                    s0_map[ix,iy,iz, ichan], t2_star_map[ix, iy, iz, ichan] = fit_t2_star(voxel_data, echo_times)
                end
            end
        end
    end
end

if combine_coils
    ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/Results/Intermediate/intermediate_image_t2star_2d", ComplexF32.(t2_star_map))
    ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/Results/Intermediate/intermediate_image_s0_2d", ComplexF32.(s0_map))
else
    ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/Results/Intermediate/intermediate_image_t2star_2d_no_combine_coils", ComplexF32.(t2_star_map))
    ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/Results/Intermediate/intermediate_image_s0_2d_no_combine_coils", ComplexF32.(s0_map))
end

function visualize_slices(t2_star_map)
    anim = @animate for (i, slice) in enumerate(eachslice(t2_star_map; dims=3))
        heatmap(slice, color=:viridis, title="Slice $i")
    end
    gif(anim, "t2_star-map.gif", fps=5)
end

visualize_slices(t2_star_map[:,:,:,1])
