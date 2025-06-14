function initialise_real_params(e_d, s0_d)
    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    # b0_d .= 0
    if !isfile("/mnt/f/Dominic/Results/B0/2d/delta_b0_dcf.cfl")
        @info "No B0 prediction detected - Generating prediction"
        real_b0_prediction()
    end
    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0_dcf"))

    im = -γ .* b0_d

    # s0_d .= 0.0;
    s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x_dcf")[:, :, :, 1])
    
    if !isfile("/mnt/f/Dominic/Results/Intermediate/2d/s0_dcf.cfl") || !isfile("/mnt/f/Dominic/Results/Intermediate/2d/t2_dcf.cfl")
        @info "No S0 or T2 prediction detected - Generating prediction"    
        real_s0_prediction()
    end

    # r2_d .= 1.0 / 50

    t2 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/t2_dcf"))
    mask = (t2 .> 0.0) .& isfinite.(t2)

    r2_d .= 0.0
    r2_d[mask] .= 1.0 ./ t2[mask]

    e_d .= complex.(r2_d, im)
    # s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/s0_dcf"));
end

function initialise_real_params(e_d, s0_fat, s0_water)
    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    # b0_d .= 0
    if !isfile("/mnt/f/Dominic/Results/B0/2d/delta_b0_dcf.cfl")
        @info "No B0 prediction detected - Generating prediction"    
        real_b0_prediction()
    end
    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0_dcf"))

    im = -γ .* b0_d

    if !isfile("/mnt/f/Dominic/Results/Intermediate/2d/s0_dcf.cfl") || !isfile("/mnt/f/Dominic/Results/Intermediate/2d/t2_dcf.cfl")
        @info "No S0 prediction detected - Generating prediction"    
        real_s0_prediction()
    end

    t2 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/t2_dcf"))
    mask = (t2 .> 0.0) .& isfinite.(t2)

    # r2_d .= 1.0 / 50

    r2_d .= 0.0
    r2_d[mask] .= 1.0 ./ t2[mask]

    e_d .= complex.(r2_d, im)

    # s0_fat .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/s0_dcf")) ./ 2
    
    # s0_fat .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x_dcf")[:, :, :, 1]) ./ 2
    # s0_water .= s0_fat
    s0_fat .= 0
    s0_water .= 0
end