function initialise_synthetic_params(eval_no, e_d, s0_d, σ)
    if !isnothing(σ)
        rounded = round(σ; digits=10)
        safe_str = replace(string(rounded), "." => "_", "-" => "m")
        σ_suffix = "_$safe_str"
    else
        σ_suffix = ""
    end

    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    t2 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2_$eval_no$σ_suffix"))
    mask = (t2 .> 0.0) .& isfinite.(t2)

    r2_d[mask] .= 1.0 ./ t2[mask]

    # r2_d .= 1.0 / 10

    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_$eval_no$σ_suffix.cfl"))

    # b0_d .= 0

    im = -γ .* b0_d
    e_d .= complex.(r2_d, im)

    s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no$σ_suffix"))

    # s0_d .= 0
end

function initialise_synthetic_params(eval_no, e_d, s0_fat, s0_water, σ)
    if !isnothing(σ)
        rounded = round(σ; digits=10)
        safe_str = replace(string(rounded), "." => "_", "-" => "m")
        σ_suffix = "_$safe_str"
    else
        σ_suffix = ""
    end

    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    t2 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2_fatmod_$eval_no$σ_suffix"))
    mask = (t2 .> 0.0) .& isfinite.(t2)

    r2_d[mask] .= 1.0 ./ t2[mask]


    # r2_d .= 1.0 / 50

    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_fatmod_$eval_no.cfl"))

    # b0_d .= 0

    im = -γ .* b0_d
    e_d .= complex.(r2_d, im)

    s0_total = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water")) 
    + ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_fat"))

    # s0_fat .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water"))
    # s0_water .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_water_$eval_no"))
    # s0_fat .= 0
    # s0_water .= 0
    s0_fat .= s0_total ./ 2
    s0_water .= s0_total ./ 2
end