function synthetic_b0_prediction(x, eval_no, use_fatmod=false; σ=nothing)

    if !isnothing(σ)
        rounded = round(σ; digits=10)
        safe_str = replace(string(rounded), "." => "_", "-" => "m")
        σ_suffix = "_$safe_str"
    else
        σ_suffix = ""
    end

    fm = use_fatmod ? "_fatmod" : ""

    if (isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0$(fm)_$eval_no$σ_suffix.cfl")
        && isfile("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/init_phase$(fm)_$eval_no$σ_suffix.cfl"))
        return ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0$(fm)_$eval_no$σ_suffix"),
            ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/init_phase$(fm)_$eval_no$σ_suffix")
    end

    @info "Generating B0 prediction"

    _, _, _, _, _, _, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true);

    time_since_last_rf = Float64.(time_since_last_rf)

    echo_times = vec(time_since_last_rf[268,:,1,1])

    # Mask values lower than a threshold of magnitude
    mag = mean(abs.(x), dims=4)   # → size (nx,ny,nz)
    mask = dropdims(mag .> 1e-7, dims=4)   # boolean mask of in-object voxels

    phases = permutedims(angle.(x), [4 1 2 3])

    #Phases size - (echo, nx, ny, nz)
    unwrap!(phases, dims=1)

    #Unwrapped size - (echo, nx * ny * nz)
    unwrapped = reshape(phases, size(phases)[1], :)

    N  = nx * ny * nz

    # Following the equation:
    # phase[S(t)] = y * Δb0 * t + phase[S(0)]
    # Fit to phase[S(t)] = a * t + b
    # Where a = y * Δb0
    # b = phase[S(0)]

    a = Vector{Float64}(undef, N)
    b = Vector{Float64}(undef, N)

    for i in 1:N
        if !mask[i]
            a[i] = 0.0
            b[i] = 0.0
            continue
        end

        p = fit(echo_times, @view(unwrapped[:,i]), 1)
        c = coeffs(p)

        # pad with zeros
        if length(c) < 2
            resize!(c, 2)
        end

        a[i] = c[1]
        b[i] = c[2]
    end

    Δb0  = reshape(a ./ γ, nx, ny, nz)
    init_phase = reshape(b, nx, ny, nz)

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0$(fm)_$eval_no$σ_suffix", ComplexF32.(Δb0))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/init_phase$(fm)_$eval_no$σ_suffix", ComplexF32.(init_phase))

    return Δb0, init_phase
end