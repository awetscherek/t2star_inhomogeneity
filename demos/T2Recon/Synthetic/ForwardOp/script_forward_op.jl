using DqT2
using FINUFFT

# Configure Settings
combine_coils = true
use_fat_modulation = false

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation)

x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

dims = [nx, ny]
tol=1e-9

(
    y_d,
    kx_d,
    ky_d,
    dcf_d,
    c_d,
    selection,
    num_timepoints,
    num_total_timepoints,
    fat_modulation
) = preprocess_data(
    config,
    raw,
    combine_coils,
    sens,
    kx,
    ky,
    timepoint_window_size,
    use_dcf,
    fat_modulation
    )

# ------------------------------------------

r2_d = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])
b0_d = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

if !isnothing(fat_modulation)
    s0_fat_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
    s0_water_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

    initialise_params(r2_d,b0_d, s0_fat_d, s0_water_d)
else
    s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

    initialise_params(r2_d,b0_d, s0_d)
end

# plan NUFFTs:
plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

r = Array{ComplexF64}(undef, size(y_d))

function forward_operator(r2, b0, fat,water)
    return forward_operator_impl(plan2, r2, b0, fat, water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
    timepoint_window_size, fat_modulation)
end

function forward_operator(r2, b0, s0)
    return forward_operator_impl(plan2, r2, b0, nothing, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
    timepoint_window_size, fat_modulation)
end

if !isnothing(fat_modulation)
    x = forward_operator(r2_d, b0_d, s0_fat_d, s0_water_d)
else
    x = forward_operator(r2_d, b0_d, s0_d)
end

println("size of x")
println(size(x))

# comb = combine_coils ? "" : "_no_combine_coils"
# fat_mod = use_fat_modulation ? "_fatmod" : ""

# ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/x_$timepoint_window_size$comb$fat_mod", ComplexF32.(x))