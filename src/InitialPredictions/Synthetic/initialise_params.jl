function initialise_synthetic_params(eval_no, e_d, s0_d)
    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    r2_d .= 1.0 / 50

    # b0_d .= 0
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_delta_b0.cfl")
        @info "No B0 prediction detected - Generating prediction"
        synthetic_b0_prediction(eval_no)
    end
    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_delta_b0.cfl"))

    im = -γ .* b0_d
    e_d .= complex.(r2_d, im)

    # s0_d .= 0.0;
    
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_s0.cfl")
        @info "No S0 prediction detected - Generating prediction"    
        synthetic_s0_prediction(eval_no)
    end
    s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_s0.cfl"));
end

function initialise_synthetic_params(eval_no, e_d, s0_fat, s0_water)
    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    r2_d .= 1.0 / 50

    # b0_d .= 0
    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_delta_b0.cfl")
        @info "No B0 prediction detected - Generating prediction"
        synthetic_b0_prediction(eval_no)
    end
    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_delta_b0.cfl"))

    im = -γ .* b0_d
    e_d .= complex.(r2_d, im)

    if !isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_s0.cfl")
        @info "No S0 prediction detected - Generating prediction"    
        synthetic_s0_prediction(eval_no)
    end
    s0_fat .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_s0.cfl")) ./ 2;
    s0_water .= s0_fat
end