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
    
    # r2_d = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])
    # b0_d = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    e_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

    if !isnothing(fat_modulation)
        s0_fat_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
        s0_water_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

        initialise_params(r2_d,b0_d, s0_fat_d, s0_water_d)
    else
        s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
        initialise_params(e_d, s0_d)
        # initialise_params(r2_d,b0_d, s0_d)
    end

    # plan NUFFTs:
    plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    r = Array{ComplexF64}(undef, size(y_d))

    function flatten(r2,b0, s0)
        s0_r, s0_i = real(s0), imag(s0)
        return vcat(vec(r2), vec(b0), vec(s0_r), vec(s0_i))
    end

    function flatten(e, s0)
        return vcat(vec(e), vec(s0))
    end

    function flatten(r2,b0, fat,water)
        fat_r, fat_i = real(fat), imag(fat)
        water_r, water_i = real(water), imag(water)
        return vcat(vec(r2), vec(b0), vec(fat_r), vec(fat_i), vec(water_r), vec(water_i))
    end

    # function unflatten(X)
    #     if !isnothing(fat_modulation)
    #         N = length(X) ÷ 6
    #         return reshape(X[1:N], size(r2_d)), reshape(X[N+1:2*N], size(b0_d)), Complex.(reshape(X[2*N+1:3*N], size(s0_fat_d)), reshape(X[3*N+1:4*N], size(s0_fat_d))), Complex.(reshape(X[4*N+1:5*N], size(s0_water_d)), reshape(X[5*N+1:end], size(s0_water_d)))
    #     else
    #         N = length(X) ÷ 4
    #         return reshape(X[1:N], size(r2_d)), reshape(X[N+1:2*N], size(b0_d)), Complex.(reshape(X[2*N+1:3*N], size(s0_d)), reshape(X[3*N+1:end], size(s0_d)))
    #     end
    # end

    function unflatten(X)
        N = length(X) ÷ 2
        return reshape(X[1:N], size(e_d)), reshape(X[N+1:end], size(s0_d))
    end

    # Initialise Operators with implicit values
    function forward_operator(e, fat, water)
        r .= forward_operator_impl(plan2, e, fat, water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)

    end

    function forward_operator(e,s0)
        return forward_operator_impl(plan2, e, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)
    end

    function adjoint_operator!(e, fat, water)
        return adjoint_operator_impl(plan1, r, e, fat, water, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
        timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, nx, ny, nz, config["nchan"])
        storage .= flatten(g_r2, g_b0, g_fat_s0, g_water_s0)
    end

    function adjoint_operator(e, s0)
        return adjoint_operator_impl(plan1, r, e, s0, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
        timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, nx, ny, nz, config["nchan"])
    end

    if !isnothing(fat_modulation)
        obj = forward_operator(e, s0_fat_d, s0_water_d)
    else
        obj = forward_operator(e_d, s0_d)
    end

    # Optimiser
    model = (S0 = s0_d, e = e_d)
    state = Optimisers.setup(Optimisers.AdamW(), model)

    for it = 1:niter        
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

    finufft_destroy!(plan1)
    finufft_destroy!(plan2)

    # Im{e} = - γ .* Δb0
    # Δb0 = - Im{e} ./ γ
    b0 = imag(e_d) ./ (-γ)

    # collect results from GPU & return:
    if !isnothing(fat_modulation) 
        1 ./ r2_d, s0_fat_d, s0_water_d, b0_d
    else
        1 ./ real(e_d), nothing, s0_d, b0
    end
end