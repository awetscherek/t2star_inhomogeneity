using FINUFFT
using Optimisers
using ProgressBars

function forward_op_synthetic_data_test(config, kx, ky, raw, timepoints, dims; # keyword arguments: 
    combine_coils=false,      # whether to use coil sensitivities
    sens=nothing,             # coil sensitivities ...
    use_dcf=false,            # whether to use pre-conditioner
    tol=1e-9,                 # tolerance for FINUFFT
    niter=use_dcf ? 10 : 100, # number of gradient descent iterations
    timepoint_window_size=536,  # number of samples within each timepoint approximation window
    fat_modulation=nothing) # consideration of fat and water

    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    #kx and ky should be of shape
    #(269,8,536)
    # nky => necho => nkx
    nx, ny = dims;
    nz = size(raw, 4) # this assumes only data from one echo is passed to the function, but several slices, so raw should be a 4D array

    nkx, _, nky, _ = size(kx)

    @assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

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

    dcf_y = use_dcf ? reshape(sqrt.(dcf), 1, size(dcf, 1), 1, 1, 1) : dcf

    y_d = reshape(ComplexF64.(permutedims(raw, [3 1 5 4 2])) .* dcf_y, config["necho"] * nkx, :, nz * config["nchan"])[selection, :]

    dcf_d = use_dcf ? repeat(sqrt.(dcf), outer=(size(ky, 2), size(ky, 3)))[selection] : 1.0

    num_total_timepoints = config["necho"] * nkx
    num_timepoints = ceil(Int, num_total_timepoints / timepoint_window_size)

    # Reshape fat modulation so it can be easily multiplied in k-space
    if !isnothing(fat_modulation)
        fat_modulation = repeat(vec(fat_modulation), 1, nky)[selection]
        fat_modulation .+= 1
    end

    γ = 2 * π * 42.576e6

    # Initial value of exponent(e) and S0:
    init_prediction_dcf = true
    ip_dcf = init_prediction_dcf ? "_dcf" : ""

    # Take initial value of S0 to be reconstruction
    # s0_d .= 0.0;
    # s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$ip_dcf")[:, :, :, 1])
    # s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$ip_dcf"));

    r2 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])
    Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    #im = im{e} = - γ .* Δb0
    im = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    Δb0 .= 0
    # Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$ip_dcf"))

    s0_d .= 1
    r2 .= 1 ./ 50.0

    im = -γ .* Δb0

    e_d .= complex.(r2, im)

    #intermediate result, required for gradient at each time point
    g_r_t = Array{ComplexF64}(undef, nx, ny, nz * config["nchan"])
    g_e = combine_coils ? Array{ComplexF64}(undef, size(e_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"])
    g_s0 = combine_coils ? Array{ComplexF64}(undef, size(s0_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"])

    # plan NUFFTs:
    plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    r = Array{ComplexF64}(undef, size(y_d))
    r .= forward_operator(plan2, e_d, s0_d, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
    timepoint_window_size, fat_modulation)

    r .*= dcf_d
    r .-= y_d

    obj = real(r[:]' * r[:]) / 2.0 # objective function

    initial_obj = "Initial obj = $obj"
    @info initial_obj

    open("output.txt", "a") do f
        println(f, string(initial_obj))
    end

    # Optimiser
    model = (S0=s0_d, e=e_d)
    state = Optimisers.setup(Optimisers.AdamW(), model)

    iter = ProgressBar(1:niter)
    for it in iter
        g_e, g_s0 = jacobian_operator(plan1, r, e_d, s0_d, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
            timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, g_r_t, fat_modulation, nx, ny, nz, config["nchan"])

        gradients = (S0=g_s0, e=g_e)
        state, model = Optimisers.update(state, model, gradients)
        s0_d, e_d = model.S0, model.e

        r .= forward_operator(plan2, e_d, s0_d, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
            timepoint_window_size, fat_modulation)

        r .*= dcf_d
        r .-= y_d

        obj = real(r[:]' * r[:]) / 2.0

        info = "it = $it, obj = $obj"
        # @info info
        open("output.txt", "a") do f
            println(f, string(info))
        end

        set_description(iter, "obj: $obj")
    end

    finufft_destroy!(plan1)
    finufft_destroy!(plan2)

    # Im{e} = - γ .* Δb0
    # Δb0 = - Im{e} ./ γ
    Δb0 = imag(e_d) ./ (-γ)

    # collect results from GPU & return: 
    1 ./ real(e_d), s0_d, Δb0
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

function jacobian_operator(plan1, r, e_d, s0_d, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
    timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, g_r_t, fat_modulation, nx, ny, nz, nchan)

    # Initialize sum of gradients (nx,ny,nz,nchan) prior to summing of gradients over coils
    g_s0_total = zeros(ComplexF64, nx, ny, nz, nchan)
    g_e_total = zeros(ComplexF64, nx, ny, nz, nchan)

    start_idx = 1
    for t in ProgressBar(1:num_timepoints)
        t_start = (t - 1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, num_total_timepoints)
        approx_t = round(Int, (t_start + t_end) / 2)

        t_ms = timepoints[approx_t]

        # Get the boolean mask for timepoint t (assume selection is 2D with one row per timepoint)
        sel = selection[t_start:t_end, :]
        npoints = sum(sel)

        # Extract the segment of the residual corresponding to timepoint t.
        r_t = r[start_idx:start_idx+npoints-1, :]
        fat_modulation_t = !isnothing(fat_modulation) ? view(fat_modulation,start_idx:start_idx+npoints-1) : 1.0

        dcf_t = use_dcf ? view(dcf_d, start_idx:start_idx+npoints-1) : 1.0

        kx_d_t = collect(kx_d[t_start:t_end, :][sel])
        ky_d_t = collect(ky_d[t_start:t_end, :][sel])

        finufft_setpts!(plan1, kx_d_t, ky_d_t)

        finufft_exec!(plan1, r_t .* dcf_t .* conj.(fat_modulation_t), g_r_t)

        if combine_coils
            g_r_result_t = reshape(g_r_t, size(c_d)) .* conj(c_d)
        else
            g_r_result_t = reshape(g_r_t, size(e_d))
        end

        conj_s0 = conj.(s0_d)
        conj_exp_term = conj.(exp.(-t_ms .* e_d))

        g_e_total .+= (-conj_s0 .* t_ms .* conj_exp_term) .* g_r_result_t
        g_s0_total .+= conj_exp_term .* g_r_result_t

        #TODO: Maybe put the sum of gradients in for loop instead of at end

        start_idx += npoints
    end

    if combine_coils
        g_e_total = dropdims(sum(g_e_total, dims=4), dims=4)
        g_s0_total = dropdims(sum(g_s0_total, dims=4), dims=4)
    end

    return g_e_total, g_s0_total
end

