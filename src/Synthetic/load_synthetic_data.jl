function load_synthetic_data(eval_no, config, combine_coils, sens, kx, ky, use_dcf, timepoints, fat_modulation, σ=nothing)

    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0.cfl"))
        @info "Generating Synthetic Data"

        phantom(eval_no)
    end

    if !isnothing(σ)
        rounded = round(σ; digits=10)
        safe_str = replace(string(rounded), "." => "_", "-" => "m")
        σ_suffix = "_$safe_str"
    else
        σ_suffix = ""
    end
    
    
    if (isfile("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no$σ_suffix.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2_$eval_no.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/s0_$eval_no.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_$eval_no.cfl"))
        return ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no$σ_suffix"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2_$eval_no"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/s0_$eval_no"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_$eval_no")
    end

    #Raw data generated with NO approximation
    timepoint_window_size = 1

    dims = [nx,ny]
    tol=1e-9 #maybe change

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
        timepoint_window_size, fat_modulation)
    end

    function split_ksp_by_echo(ksp)
        #Split into echos
        echo_splits = [142730, 142699, 142700, 142698, 142701, 142698, 142701, 142697]
        offsets = cumsum([0; echo_splits])

        return [
            @view ksp[offsets[i]+1:offsets[i+1],:]
            for i in 1:length(echo_splits)
        ]
    end

    x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no$σ_suffix.cfl")
        || !isfile("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_$eval_no$σ_suffix.cfl"))

        @info "Raw data for Evaluation $eval_no fatmod not found for σ = $(isnothing(σ) ? 0 : σ) - Generating:"

        r2 = 1 ./ Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2"))
        s0 = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0"))
        b0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0"))

        im = -γ .* b0
        e = complex.(r2, im)

        if isfile("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no.cfl")
            y_d = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no")

            if !isnothing(σ)
                y_d = add_kspace_noise(y_d, σ)
            #     ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no$σ_suffix", ComplexF32.(y_d))
            end
        else
            y_d = synth_recon_forward_operator(e, s0)
            ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no", ComplexF32.(y_d))

            if !isnothing(σ)
                y_d = add_kspace_noise(y_d, σ)
                # ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no$σ_suffix", ComplexF32.(y_d))
            end
        end
        
        y = split_ksp_by_echo(y_d)

        if combine_coils
            c_d = calculate_synthetic_coil_sensitivity(config)
        end


        @info "Generating Image Reconstruction of Synthetic Raw Data"
        #Reconstructing Synthetic data for Intermediate Image Initial Prediction
        for (ie, xe) in zip(1:config["necho"], eachslice(x, dims=length(size(x))))
            xe .= image_recon_synthetic_2d(config, 
            @view(kx[:, ie, :, :]),
            @view(ky[:, ie, :, :]),
            y[ie],
            combine_coils = combine_coils,
            sens = c_d,
            use_dcf = use_dcf,
            )
        end
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_$eval_no$σ_suffix", ComplexF32.(x)) 
    else
        y_d = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_$eval_no$σ_suffix")
        x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_$eval_no$σ_suffix")
    end

    #Generate Initial Predictions

    # We use the first echo reconstruction as an initial guess
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no$σ_suffix.cfl")
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no$σ_suffix", ComplexF32.(x[:,:,:,1]))
    end

    # Initial Guess of B0
    b0, s0_phase = synthetic_b0_prediction(x, eval_no, σ=σ)

    #Generate an Intermediate Image reconstruction of the T2
    intermediate_t2, intermediate_s0 = generate_intermediate_image_prediction(x, b0, s0_phase, eval_no, σ=σ)
    
    return y_d, intermediate_t2, intermediate_s0, b0
end


#-------------------------------------------------------
#FatMod

function load_synthetic_data_fatmod(eval_no, config, combine_coils, sens, kx, ky, use_dcf, timepoints, fat_modulation)

    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2_fatmod.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_fat.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0_fatmod.cfl"))
        @info "Generating Synthetic Data"

        phantom_fatmod(eval_no)
    end
    
    if (isfile("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_fatmod_$eval_no.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2_fatmod_$eval_no.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/s0_fatmod_$eval_no.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_fatmod_$eval_no.cfl"))
        return ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_fatmod_$eval_no"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2_fatmod_$eval_no"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/s0_fatmod_$eval_no"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_fatmod_$eval_no")

    end

    #Raw data generated with NO approximation
    timepoint_window_size = 536

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

    function synth_recon_forward_operator_fatmod(e,fat, water)
        return forward_operator_impl(plan2, e, fat, water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)
    end

    function split_ksp_by_echo(ksp)
        #Split into echos
        echo_splits = [142730, 142699, 142700, 142698, 142701, 142698, 142701, 142697]
        offsets = cumsum([0; echo_splits])

        return [
            @view ksp[offsets[i]+1:offsets[i+1],:]
            for i in 1:length(echo_splits)
        ]
    end

    x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_fatmod_$eval_no.cfl")
        || !isfile("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_fatmod_$eval_no.cfl"))

        @info "Raw data for Evaluation $eval_no fatmod not found - Generating:"

        r2 = 1 ./ Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2_fatmod"))
        fat = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_fat"))
        water = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water"))
        b0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0_fatmod"))

        im = -γ .* b0
        e = complex.(r2, im)

        y_d = synth_recon_forward_operator_fatmod(e, fat, water)
        
        y = split_ksp_by_echo(y_d)

        ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_fatmod_$eval_no", ComplexF32.(y_d))

        if combine_coils
            c_d = calculate_synthetic_coil_sensitivity(config)
        end


        @info "Generating Image Reconstruction of Synthetic Raw Data"
        #Reconstructing Synthetic data for Intermediate Image Initial Prediction
        for (ie, xe) in zip(1:config["necho"], eachslice(x, dims=length(size(x))))
            xe .= image_recon_synthetic_2d(config, 
            @view(kx[:, ie, :, :]),
            @view(ky[:, ie, :, :]),
            y[ie],
            combine_coils = combine_coils,
            sens = c_d,
            use_dcf = use_dcf,
            )
        end
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_fatmod_$eval_no", ComplexF32.(x)) 
    else
        y_d = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/RawData/y_d_fatmod_$eval_no")
        x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/ImageRecon/synth_recon_fatmod_$eval_no")
    end

    #Generate Initial Predictions

    # We use the first echo reconstruction as an initial guess
    if (!isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_fat_$eval_no.cfl")
        || !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_water_$eval_no.cfl")
        )
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_fat_$eval_no", ComplexF32.(x[:,:,:,1] ./ 2))
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_water_$eval_no", ComplexF32.(x[:,:,:,1] ./ 2))
    end

    # Initial Guess of B0
    b0, s0_phase = synthetic_b0_prediction(x, eval_no, true)

    #Generate an Intermediate Image reconstruction of the T2
    intermediate_t2, intermediate_s0 = generate_intermediate_image_prediction(x, b0, s0_phase, eval_no, true)
    
    return y_d, intermediate_t2, intermediate_s0, b0
end


function phantom(x)
    name = Symbol("gen_phantom_$x")
    f = getfield(@__MODULE__, name)
    return f(x)
end

function phantom_fatmod(x)
    name = Symbol("gen_phantom_fatmod_$x")
    f = getfield(@__MODULE__, name)
    return f(x)
end

function add_kspace_noise(ksp, σ)

    if isnothing(σ) || σ == 0
        return ksp
    end

    noise_real = σ * randn(eltype(real(ksp)), size(ksp))
    noise_imag = σ * randn(eltype(real(ksp)), size(ksp))
    return ksp .+ complex.(noise_real, noise_imag)
end