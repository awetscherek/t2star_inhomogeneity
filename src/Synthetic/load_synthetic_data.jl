function load_synthetic_data(eval_no, config, combine_coils, sens, kx, ky, use_dcf, timepoints, fat_modulation)

    if isfile("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no.cfl")
        return ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2_$eval_no")
    end

    @info "Raw data for Evaluation $eval_no fatmod not found - Generating:"

    #Raw data generated with NO approximation
    timepoint_window_size = 1

    dims = [nx,ny]
    tol=1e-9

    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)

    (
        _,
        kx_d,
        ky_d,
        _,
        c_d,
        selection,
        num_timepoints,
        num_total_timepoints,
        fat_modulation
    ) = preprocess_data(
        config,
        nothing,
        combine_coils,
        sens,
        kx,
        ky,
        timepoint_window_size,
        use_dcf,
        fat_modulation,
        true
        )

    function synth_recon_forward_operator(e,s0)
        return forward_operator_impl(plan2, e, nothing, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation, true)
    end

    #function modified later to add noise to kspace
    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0.cfl"))
        @info "Generating Synthetic Data"

        phantom(eval_no, config["nchan"])
    end

    r2 = 1 ./ Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2"))
    s0 = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0"))
    b0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0"))

    im = -γ .* b0
    e = complex.(r2, im)

    y = synth_recon_forward_operator(e, s0)
    
    y_d = vcat(y...)

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no", ComplexF32.(y_d))

    time_step = ceil(Int, size(kx)[1] / timepoint_window_size)

    if combine_coils
        c_d = calculate_synthetic_coil_sensitivity(config)
    end

    x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

    #Create an Image reconstruction of the Raw data 
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/synth_recon_$eval_no.cfl")
        @info "Generating Image Reconstruction of Synthetic Raw Data"
        #Reconstructing Synthetic data for Intermediate Image Initial Prediction
        for (ie, xe) in zip(1:config["necho"], eachslice(x, dims=length(size(x))))
            xe .= image_recon_synthetic_2d(config, 
            @view(kx[:, ie, :, :]),
            @view(ky[:, ie, :, :]),
            vcat(y[(ie-1)*time_step + 1:ie*time_step]...),
            combine_coils = combine_coils,
            sens = c_d,
            use_dcf = use_dcf,
            )
        end
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_$eval_no", ComplexF32.(x))
    else
        x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_$eval_no")
    end

    #Generate Initial Predictions

    # We use the first echo reconstruction as an initial guess
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no.cfl")
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no", ComplexF32.(x[:,:,:,1]))
    end

    # Initial Guess of B0
    b0, s0_phase = synthetic_b0_prediction(x, eval_no)

    #Generate an Intermediate Image reconstruction of the T2
    intermediate_t2 = generate_intermediate_image_prediction(x, b0, s0_phase, eval_no)
    
    return y_d, intermediate_t2
end

function load_synthetic_data_fatmod(eval_no, config, combine_coils, sens, kx, ky, use_dcf, fat_modulation)

    if isfile("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_fatmod_$eval_no.cfl")
        return ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_fatmod_$eval_no.cfl")
    end

    @info "Raw data for Evaluation $eval_no fatmod not found - Generating:"

    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)

    (
        _,
        kx_d,
        ky_d,
        _,
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
        1, #Timepoint_window_size 
        use_dcf,
        fat_modulation,
        true
        )

    function synth_recon_forward_operator_fatmod(e, fat, water)
        return forward_operator_impl(plan2, e, fat, water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation, true)
    end

    #function modified later to add noise to kspace
    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_fat.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0.cfl"))
        @info "Generating Synthetic Data"

        phantom_fatmod(eval_no, nchan) ## change
    end

    r2 = 1 ./ Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2"))
    fat = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_fat"))
    water = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water"))
    b0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0"))

    im = -γ .* b0
    e = complex.(r2, im)

    y = synth_recon_forward_operator_fatmod(e, fat, water)

    time_step = ceil(Int, size(kx)[1] / timepoint_window_size)

    if combine_coils
        c_d = calculate_synthetic_coil_sensitivity(config)
    end

    x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

    #Create an Image reconstruction of the Raw data 
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/synth_recon_$eval_no.cfl")
        #Reconstructing Synthetic data for Intermediate Image Initial Prediction
        for (ie, xe) in zip(1:config["necho"], eachslice(x, dims=length(size(x))))
            xe .= image_recon_synthetic_2d(config, 
            @view(kx[:, ie, :, :]),
            @view(ky[:, ie, :, :]),
            vcat(y[(ie-1)*time_step + 1:ie*time_step]...),
            combine_coils = combine_coils,
            sens = c_d,
            use_dcf = use_dcf,
            )
        end
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_$eval_no", ComplexF32.(x))
    else
        x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_$eval_no")
    end

    #Generate Initial Predictions

    # We use the first echo reconstruction as an initial guess
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no.cfl")
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no", ComplexF32.(x[:,:,:,1]))
    end
    
    # Initial Guess of B0
    b0, _ = synthetic_b0_prediction(x, eval_no)

    #No intermediate image prediction - Not capable of fatmod
    return vcat(y...), nothing
end

function phantom(x,a)
    name = Symbol("gen_phantom_$x")
    f = getfield(@__MODULE__, name)
    return f(x,a)
end

function phantom_fatmod(x,a)
    name = Symbol("gen_phantom_fatmod_$x")
    f = getfield(@__MODULE__, name)
    return f(x,a)
end