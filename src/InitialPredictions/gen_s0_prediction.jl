"""
    fit_t2star(x, echo_times, b0_map, s0_phase; mask=nothing)

Fit T2* and S0 magnitude from complex MRI data.

# Arguments
- `x`: Complex signal reconstruction of shape (nx, ny, nz, necho)
- `echo_times`: Echo times in seconds (array of length necho)
- `b0_map`: B0 field map in Tesla (shape: nx, ny, nz)
- `s0_phase`: Phase of S0 (shape: nx, ny, nz)
- `mask`: Optional binary mask to restrict fitting (shape: nx, ny, nz)

# Returns
- `t2star_map`: T2* map in seconds (shape: nx, ny, nz)
- `s0_mag`: S0 magnitude (shape: nx, ny, nz)
"""
function fit_t2star(x, echo_times, b0_map, s0_phase; mask=nothing)
    # Initialize output maps
    t2star_map = zeros(Float64, nx, ny, nz)
    s0_mag = zeros(Float64, nx, ny, nz)
    
    # Create a default mask if none provided
    if isnothing(mask)
        mask = trues(nx, ny, nz)
    end
    
    # Prepare for fitting
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                if !mask[i,j,k]
                    continue
                end
                
                # Extract the signal time course for this voxel
                signal_timecourse = vec(x[i,j,k,:])
                
                # Extract B0 and S0 phase for this voxel
                b0_val = b0_map[i,j,k]
                phase_val = s0_phase[i,j,k]
                
                # Remove the B0-induced phase evolution and initial phase
                corrected_signal = signal_timecourse .* exp.(-im * (γ * b0_val .* echo_times .+ phase_val))
                
                # Now we have |S₀|e^(-TE/T2*), take magnitude
                magnitude_data = abs.(corrected_signal)
                
                # Fit mono-exponential to magnitude data
                t2star_val, s0_mag_val = fit_monoexponential(echo_times, magnitude_data)
                
                # Store results
                t2star_map[i,j,k] = t2star_val
                s0_mag[i,j,k] = s0_mag_val
            end
        end
    end
    
    return t2star_map, s0_mag
end

"""
    fit_monoexponential(xdata, ydata)

Fit a mono-exponential decay function: y = a * exp(-x/b)

# Arguments
- `xdata`: Independent variable (time)
- `ydata`: Dependent variable (signal magnitude)

# Returns
- `b`: Time constant (T2*)
- `a`: Initial amplitude (|S₀|)
"""
function fit_monoexponential(xdata, ydata)
    # Handle case of low SNR or invalid data
    if any(isnan.(ydata)) || any(isinf.(ydata)) || maximum(ydata) < eps()
        return 0.0, 0.0
    end
    
    # Normalize data for better numerical stability
    scale_factor = maximum(ydata)
    y_normalized = ydata ./ scale_factor
    
    # Linear fit using log transform
    # log(y) = log(a) - x/b
    valid_indices = findall(y_normalized .> eps())
    
    if length(valid_indices) < 2
        return 0.0, 0.0
    end
    
    x_valid = xdata[valid_indices]
    y_valid = log.(y_normalized[valid_indices])
    
    # Simple linear regression
    # y = p[1] + p[2]*x where p[1] = log(a) and p[2] = -1/b
    X = hcat(ones(length(x_valid)), x_valid)
    p = X \ y_valid
    
    # Extract parameters
    log_a = p[1]
    neg_inv_b = p[2]
    
    # Convert back to physical parameters
    a = exp(log_a) * scale_factor
    
    # Handle case where slope is zero or positive (non-physical)
    if neg_inv_b >= 0
        b = Inf
    else
        b = -1.0 / neg_inv_b
    end

    b = min(b, 0.2)
    
    return b, a  # Return T2* and |S₀|
end

function gen_s0_prediction()
    config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true)

    combine_coils = true
    use_dcf = true

    x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);
    Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

    # mag_s0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
    # ϕs0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
    # s0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

    echo_times = time_since_last_rf[268,:,1,1]
    # echo_times .*= 1e-3

    # println("Echo times: ", echo_times)

    comb = combine_coils ? "" : "_no_combine_coils"
    dcf = use_dcf ? "_dcf" : ""

    x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$comb$dcf")
    Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$comb$dcf"))
    #Phase in degrees
    s0_phase = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/init_phase$comb$dcf"))

    println("Data Loaded")

    t2, s0_mag = fit_t2star(x, echo_times, Δb0, s0_phase)

    s0 = s0_mag .* cis.((s0_phase .* (π/180)))

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/t2$comb$dcf", ComplexF32.(t2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$comb$dcf", ComplexF32.(s0))
end