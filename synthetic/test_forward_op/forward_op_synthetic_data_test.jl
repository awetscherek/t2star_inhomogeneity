using FINUFFT
using Optimisers
using ProgressBars

function forward_op_synthetic_data_test(config, kx, ky, timepoints, dims; # keyword arguments: 
    combine_coils=false,      # whether to use coil sensitivities
    sens=nothing,             # coil sensitivities ...
    tol=1e-9,                 # tolerance for FINUFFT
    timepoint_window_size=536,  # number of samples within each timepoint approximation window
    fat_modulation=nothing) # consideration of fat and water

    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    #kx and ky should be of shape
    #(269,8,536)
    # nky => necho => nkx
    nx, ny, nz = dims;

    nkx, _, nky, _ = size(kx)

    @assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

    c_d = combine_coils ? sens : [1.0] # this shouldn't make a copy of sens

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [2 1 3 4]) * config["FOVx"] / nx * 2 * pi, :, nky)
    ky_d = reshape(permutedims(ky, [2 1 3 4]) * config["FOVy"] / ny * 2 * pi, :, nky)

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi

    # Considering the phase equation:
    # S(t) = S(0) .* exp(i .* γ .* Δb0 .* t - (t / T2*) )
    # Consider the exponent (t / T2*) - i .* γ .* Δb0 .* t
    # Such that S(t) = S(0) .* exp(e_d)
    # We use a variable e such that
    # Real{e} = (1 / T2*)
    # Im{e} = - γ .* Δb0
    # e = (1/T2* - i .* γ .* Δb0)
    # Then, exp(- t * e) = exp(i .* γ .* Δb0 .* t - (t / T2*) )

    e_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
    s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

    # this is the raw data from which we want to reconstruct the coil images
    #(num_timepoints, ky, nz * nchan)

    num_total_timepoints = config["necho"] * nkx
    num_timepoints = ceil(Int, num_total_timepoints / timepoint_window_size)

    # Reshape fat modulation so it can be easily multiplied in k-space
    if !isnothing(fat_modulation)
        fat_modulation = repeat(vec(fat_modulation), 1, nky)[selection]
        fat_modulation .+= 1
    end

    γ = 2 * π * 42.576e6

    r2 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])
    Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    #im = im{e} = - γ .* Δb0
    im = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    # Initial value of exponent(e) and S0:
    init_prediction_dcf = true
    ip_dcf = init_prediction_dcf ? "_dcf" : ""

    # Take initial value of S0 to be reconstruction
    # s0_d .= 0.0;
    # s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$ip_dcf")[:, :, :, 1])
    # s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$ip_dcf"));

    Δb0 .= 0
    # Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$ip_dcf"))

    s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/s0_1"))
    r2 .= 1 ./ Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/t2_50"))

    im = -γ .* Δb0

    e_d .= complex.(r2, im)

    # plan NUFFTs:
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    y = forward_operator(plan2, e_d, s0_d, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
    timepoint_window_size, fat_modulation)

    finufft_destroy!(plan2)

    # collect results from GPU & return: 
    y
end

function forward_operator(plan2, e_d, s0_d, num_timepoints, num_total_timepoints, kx_d, ky_d,
    c_d, timepoints, selection, timepoint_window_size, fat_modulation)
    y_list = Vector{Array{ComplexF64}}(undef, num_timepoints)
    for t in ProgressBar(1:num_timepoints)
        t_start = (t - 1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, num_total_timepoints)
        approx_t = round(Int, (t_start + t_end) / 2)

        t_ms = timepoints[approx_t]

        sel = selection[t_start:t_end, :]
        kx_d_t = collect(kx_d[t_start:t_end, :][sel])
        ky_d_t = collect(ky_d[t_start:t_end, :][sel])

        finufft_setpts!(plan2, kx_d_t, ky_d_t)

        w_d_t = s0_d .* exp.(-t_ms .* e_d)

        y_t = finufft_exec(plan2, w_d_t .* c_d)

        y_list[t] = y_t
    end
    y = vcat(y_list...)

    if !isnothing(fat_modulation)
        y .*= fat_modulation
    end

    return y
end