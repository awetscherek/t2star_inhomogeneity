function synthetic_s0_prediction(eval_no)
    config, _, _, _, _, _, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true)

    combine_coils = true
    use_dcf = true

    x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);
    Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
    s0_phase = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

    echo_times = time_since_last_rf[268,:,1,1]
    # echo_times .*= 1e-3

    # println("Echo times: ", echo_times)

    x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/$(eval_no)_synth_recon")
    Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_delta_b0"))
    #Phase in degrees
    s0_phase .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/InitialPrediction/$(eval_no)_init_phase"))

    println("Data Loaded")

    t2, s0_mag = fit_t2star(x, echo_times, Δb0, s0_phase)

    s0 = s0_mag .* cis.((s0_phase .* (π/180)))

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_t2", ComplexF32.(t2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/InitialPrediction/$(eval_no)_s0", ComplexF32.(s0))
end