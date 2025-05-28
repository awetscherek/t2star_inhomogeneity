function generate_intermediate_image_prediction(x, b0, s0_phase, eval_no, use_fatmod=false; σ=nothing)

    if !isnothing(σ)
        rounded = round(σ; digits=10)
        safe_str = replace(string(rounded), "." => "_", "-" => "m")
        σ_suffix = "_$safe_str"
    else
        σ_suffix = ""
    end

    fm = use_fatmod ? "_fatmod" : ""

    if (isfile("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2$(fm)_$eval_no$σ_suffix.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/s0$(fm)_$eval_no$σ_suffix.cfl"))
        return ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2$(fm)_$eval_no$σ_suffix"), 
                ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/s0$(fm)_$eval_no$σ_suffix")
    end

    @info "Generating Intermediate Image Prediction"

    _, _, _, _, _, _, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true)

    echo_times = time_since_last_rf[268,:,1,1]


    t2, s0_mag = fit_t2star(x, echo_times, b0, s0_phase)

    s0 = s0_mag .* cis.((s0_phase .* (π/180)))

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/t2$(fm)_$eval_no$σ_suffix", ComplexF32.(t2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/IntermediateImage/s0$(fm)_$eval_no$σ_suffix", ComplexF32.(s0))
    return t2, s0
end