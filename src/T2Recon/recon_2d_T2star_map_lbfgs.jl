function recon_2d_t2star_map(config, kx, ky, raw, timepoints, dims, ::Lbfgs; # keyword arguments: 
    combine_coils=false,      # whether to use coil sensitivities
    sens=nothing,             # coil sensitivities ...
    use_dcf=false,            # whether to use pre-conditioner
    tol=1e-9,                 # tolerance for FINUFFT
    niter=use_dcf ? 10 : 100, # number of gradient descent iterations
    timepoint_window_size=536,  # number of samples within each timepoint approximation window
    fat_modulation=nothing,
    use_synthetic=false,
    eval_no = 0) # consideration of fat and water

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
        fat_modulation,
        use_synthetic
        )

    # ------------------------------------------
    e_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

    if !isnothing(fat_modulation)
        s0_fat_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
        s0_water_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
    else
        s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
    end

    if use_synthetic
        if !isnothing(fat_modulation)
            initialise_params(Synthetic(), eval_no, e_d, s0_fat_d, s0_water_d)
        else
            initialise_params(Synthetic(), eval_no, e_d, s0_d)
        end
    else
        if !isnothing(fat_modulation)
            initialise_params(Real(), e_d, s0_fat_d, s0_water_d)
        else
            initialise_params(Real(), e_d, s0_d)
        end
    end

    # plan NUFFTs:
    plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    r = Array{ComplexF64}(undef, size(y_d))

    function flatten(e, s0)
        return vcat(vec(e), vec(s0))
    end

    function flatten(e, fat, water)
        return vcat(vec(e), vec(fat), vec(water))
    end

    function unflatten(X)
        N = length(X) ÷ 2
        return reshape(X[1:N], size(e_d)), reshape(X[N+1:end], size(s0_d))
    end

    function unflatten_fatmod(X)
        N = length(X) ÷ 3
        return reshape(X[1:N], size(e_d)), reshape(X[N+1:2*N], size(s0_fat_d)), reshape(X[2*N+1:end], size(s0_water_d))
    end

    # Initialise Operators with implicit values
    function forward_operator_fatmod(x)
        e, fat,water = unflatten_fatmod(x)
        r .= forward_operator_impl(plan2, e, fat, water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)
        r .*= dcf_d
        r .-= y_d
        obj = 1/2 * sum(abs2, r)
        @info "obj = $obj"
        return obj
    end

    function forward_operator(x)
        e, s0 = unflatten(x)
        r .= forward_operator_impl(plan2, e, nothing, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)
        r .*= dcf_d
        r .-= y_d
        obj = 1/2 * sum(abs2, r)
        @info "obj = $obj"
        return obj
    end

    function adjoint_operator!(storage, x)
        e, s0 = unflatten(x)
        g_e, g_s0 = adjoint_operator_impl(plan1, r, e, s0, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
        timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, config["nchan"])
        storage .= flatten(g_e, g_s0)
    end

    function adjoint_operator_fatmod!(storage, x)
        e, fat,water = unflatten_fatmod(x)
        g_e, g_fat, g_water = adjoint_operator_impl(plan1, r, e, fat, water, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
        timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, config["nchan"])
        storage .= flatten(g_e, g_fat, g_water)
    end

    if !isnothing(fat_modulation)
        obj = forward_operator_fatmod(flatten(e_d, s0_fat_d, s0_water_d))
    else
        obj = forward_operator(flatten(e_d, s0_d))
    end
    info="Initial Objective: $obj"
    @info info
    open("output.txt", "a") do f
        println(f, string(info))
    end

    if !isnothing(fat_modulation)
        initial_guess = flatten(e_d, s0_fat_d, s0_water_d)
    else
        initial_guess = flatten(e_d, s0_d)
    end

    if !isnothing(fat_modulation)
        results = optimize(forward_operator_fatmod, adjoint_operator_fatmod!,
            initial_guess,
            LBFGS(),
            Optim.Options(
                show_trace=true,
                iterations = niter))
    else
        results = optimize(forward_operator, adjoint_operator!,
            initial_guess,
            LBFGS(),
            Optim.Options(
                show_trace=true,
                iterations = niter))
    end

    x = Optim.minimizer(results)

    if !isnothing(fat_modulation)
        e_d, s0_fat_d, s0_water_d = unflatten_fatmod(x)
    else
        e_d, s0_d = unflatten(x)
    end

    finufft_destroy!(plan1)
    finufft_destroy!(plan2)

    # Im{e} = - γ .* Δb0
    # Δb0 = - Im{e} ./ γ
    b0 = imag(e_d) ./ (-γ)

    # collect results from GPU & return:
    if !isnothing(fat_modulation) 
        1 ./ real(e_d), s0_fat_d, s0_water_d, b0
    else
        1 ./ real(e_d), nothing, s0_d, b0
    end
end