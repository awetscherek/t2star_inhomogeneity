function gen_s0_prediction()
    config, _, _, _, _, _, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true)

    combine_coils = true
    use_dcf = true

    x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);
    Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

    echo_times = time_since_last_rf[268,:,1,1]
    # echo_times .*= 1e-3

    # println("Echo times: ", echo_times)

    comb = combine_coils ? "" : "_no_combine_coils"
    dcf = use_dcf ? "_dcf" : ""

    x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$comb$dcf")
    Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$comb$dcf"))
    phi0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/init_phase$comb$dcf"))

    println("Data Loaded")

    #---

    """
        estimate_R2star_and_S0_nls(x, echo_times, B0, phi0;)

    Same signature as before, but fits |S₀| and R2* by non-linear least squares per voxel.
    """
    function estimate_R2star_and_S0(x::Array{ComplexF64,4},
                                        echo_times::AbstractVector,
                                        B0::Array{Float64,3},
                                        phi0::Array{Float64,3})

        nx, ny, nz, necho = size(x)
        @assert length(echo_times) == necho
        @assert size(B0)   == (nx,ny,nz)
        @assert size(phi0) == (nx,ny,nz)

        # model: p = [S0mag, R2star],  y = p[1] * exp(-p[2] * t)
        model(p, t) = @. p[1] * exp(-p[2] * t)

        R2star = Array{Float64}(undef, nx,ny,nz)
        S0mag   = Array{Float64}(undef, nx,ny,nz)

        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            s = view(x, i, j, k, :)
            φ = phi0[i,j,k]
            B = B0[i,j,k]

            # phase correction
            corr = @. s * exp(-1im*(φ + γ*B*echo_times))

            # take magnitude (real+imag noise)
            ydata = abs.(corr)

            # skip if all zeros or negative nonsense
            if all(ydata .<= 0)
                R2star[i,j,k] = NaN
                S0mag[i,j,k]   = NaN
                continue
            end

            # initial guess: S0 ≈ first echo, R2* ≈ 50 s⁻¹
            p0 = [maximum(ydata), 20.0]
            try
                fit = curve_fit(model, echo_times, ydata, p0)
                p = coef(fit)
                S0mag[i,j,k]   = p[1]
                R2star[i,j,k] = p[2]
            catch e
                # if the fit fails, mark as NaN
                R2star[i,j,k] = NaN
                S0mag[i,j,k]   = NaN
            end
        end

        return R2star, S0mag
    end

    r2, s0mag = estimate_R2star_and_S0(x,echo_times, Δb0, phi0)

    s0 = s0mag .* cis.(phi0)

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/t2$comb$dcf", ComplexF32.(1.0 ./ r2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$comb$dcf", ComplexF32.(s0))
end