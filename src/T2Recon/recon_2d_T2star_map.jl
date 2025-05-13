function recon_2d_t2star_map(config, kx, ky, raw, timepoints, dims; # keyword arguments: 
    combine_coils=false,      # whether to use coil sensitivities
    sens=nothing,             # coil sensitivities ...
    use_dcf=false,            # whether to use pre-conditioner
    tol=1e-9,                 # tolerance for FINUFFT
    niter=use_dcf ? 10 : 100, # number of gradient descent iterations
    timepoint_window_size=536,  # number of samples within each timepoint approximation window
    fat_modulation=nothing) # consideration of fat and water

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
    e_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

    if !isnothing(fat_modulation)
        s0_fat_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
        s0_water_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

        initialise_params(e_d, s0_fat_d, s0_water_d)
    else
        s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
        initialise_params(e_d, s0_d)
    end

    # plan NUFFTs:
    plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    r = Array{ComplexF64}(undef, size(y_d))

    # Initialise Operators with implicit values
    function forward_operator(e, fat, water)
        return forward_operator_impl(plan2, e, fat, water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)
    end

    function forward_operator(e,s0)
        return forward_operator_impl(plan2, e, nothing, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)
    end

    function adjoint_operator(e, fat, water)
        return adjoint_operator_impl(plan1, r, e, fat, water, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
        timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, config["nchan"])
        storage .= flatten(g_r2, g_b0, g_fat_s0, g_water_s0)
    end

    function adjoint_operator(e, s0)
        return adjoint_operator_impl(plan1, r, e, s0, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
        timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, config["nchan"])
    end

    if !isnothing(fat_modulation)
        r .= forward_operator(e_d, s0_fat_d, s0_water_d)
    else
        r .= forward_operator(e_d, s0_d)
    end
    r .*= dcf_d
    r .-= y_d
    obj = 1/2 * sum(abs2, r)

    info="Initial Objective: $obj"
    @info info
    open("output.txt", "a") do f
        println(f, string(info))
    end

    # Optimiser
    if !isnothing(fat_modulation)
        model = (fat = s0_fat_d, water = s0_water_d, e = e_d)
    else
        model = (S0 = s0_d, e = e_d)
    end
    state = Optimisers.setup(Optimisers.AdamW(couple=false), model)

    for it = 1:niter
        if !isnothing(fat_modulation)
            g_e, g_fat, g_water = adjoint_operator(e_d, s0_fat_d, s0_water_d)

            gradients = (fat = g_fat, water = g_water, e = g_e)
            state, model = Optimisers.update(state, model, gradients)
            e_d, s0_fat_d, s0_water_d = model.e, model.fat, model.water

            r .= forward_operator(e_d, s0_fat_d, s0_water_d)
            r .*= dcf_d
            r .-= y_d
            obj = 1/2 * sum(abs2, r)

            info="it = $it, obj = $obj"
            @info info
            open("output.txt", "a") do f
                println(f, string(info))
            end
        else     
            g_e, g_s0 = adjoint_operator(e_d, s0_d)

            gradients = (S0 = g_s0, e = g_e)
            state, model = Optimisers.update(state, model, gradients)
            s0_d, e_d = model.S0, model.e

            r .= forward_operator(e_d, s0_d)
            r .*= dcf_d
            r .-= y_d
            obj = 1/2 * sum(abs2, r)

            info="it = $it, obj = $obj"
            @info info
            open("output.txt", "a") do f
                println(f, string(info))
            end
        end
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