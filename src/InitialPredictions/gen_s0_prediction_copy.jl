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
    # Get dimensions
    nx, ny, nz, necho = size(x)
    
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
                
                try
                    # Extract the signal time course for this voxel
                    signal_timecourse = vec(x[i,j,k,:])
                    
                    # Check for valid signal
                    if all(abs.(signal_timecourse) .< eps())
                        continue  # Skip voxels with no signal
                    end
                    
                    # Extract B0 and S0 phase for this voxel
                    b0_val = b0_map[i,j,k]
                    phase_val = s0_phase[i,j,k]
                    
                    # Handle non-finite B0 or phase values
                    if !isfinite(b0_val) || !isfinite(phase_val)
                        # Fall back to basic fit without phase correction
                        magnitude_data = abs.(signal_timecourse)
                    else
                        # Remove the B0-induced phase evolution and initial phase
                        corrected_signal = signal_timecourse .* exp.(-im * (γ * b0_val .* echo_times .+ phase_val))
                        
                        # Now we have |S₀|e^(-TE/T2*), take magnitude
                        magnitude_data = abs.(corrected_signal)
                    end
                    
                    # Skip if no valid data after correction
                    if any(!isfinite.(magnitude_data)) || all(magnitude_data .< eps())
                        continue
                    end
                    
                    # Fit mono-exponential to magnitude data
                    t2star_val, s0_mag_val = fit_monoexponential(echo_times, magnitude_data)
                    
                    # Store results
                    t2star_map[i,j,k] = t2star_val
                    s0_mag[i,j,k] = s0_mag_val
                catch e
                    # If anything fails, set default values and continue
                    t2star_map[i,j,k] = 0.0
                    s0_mag[i,j,k] = 0.0
                    # println("Error at voxel ($i,$j,$k): ", e)
                end
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
    if any(isnan.(ydata)) || any(isinf.(ydata))
        return 0.0, 0.0
    end
    
    # Normalize data for better numerical stability
    scale_factor = maximum(ydata)
    y_normalized = ydata ./ scale_factor
    
    # Linear fit using log transform
    # log(y) = log(a) - x/b
    # Only use data points with positive values
    valid_indices = findall(y_normalized .> eps())
    
    if length(valid_indices) < 2
        return 0.0, 0.0
    end
    
    x_valid = xdata[valid_indices]
    y_valid = y_normalized[valid_indices]
    
    # Further check for numerical stability - filter out very low values
    # that might cause unstable log calculations
    filtered_indices = findall(y_valid .> 0.01 * maximum(y_valid))
    
    if length(filtered_indices) < 2
        filtered_indices = 1:length(y_valid)  # If filtering removes too much, use all valid points
    end
    
    x_filtered = x_valid[filtered_indices]
    y_filtered = y_valid[filtered_indices]
    
    # Take logarithm - check for any values that might have become zero due to numerical precision
    log_y = log.(max.(y_filtered, eps()))
    
    # Simple linear regression
    # y = p[1] + p[2]*x where p[1] = log(a) and p[2] = -1/b
    X = hcat(ones(length(x_filtered)), x_filtered)
    
    # Try-catch to handle potential linear algebra errors
    try
        # Use pinv for more stable solution instead of backslash
        p = pinv(X) * log_y
        
        # Extract parameters
        log_a = p[1]
        neg_inv_b = p[2]
        
        # Convert back to physical parameters
        a = exp(log_a) * scale_factor
        
        # Handle case where slope is zero or positive (non-physical)
        if neg_inv_b >= 0
            return 0.0, scale_factor  # Return max intensity as S0 and T2*=0 for non-decaying signals
        else
            b = -1.0 / neg_inv_b
            
            # Safety limits on fitted values
            if !isfinite(b) || b <= 0
                return 0.0, 0.0
            end
            
            # Upper limit on T2* to avoid unrealistic values
            b = min(b, 1000.0)  # Cap at 1 second
            
            return b, a  # Return T2* and |S₀|
        end
    catch e
        # Fall back to robust estimation if linear regression fails
        # Simple two-point estimate between first and last points
        if length(x_filtered) >= 2
            first_idx = 1
            last_idx = length(x_filtered)
            
            # Calculate slope between first and last points
            delta_x = x_filtered[last_idx] - x_filtered[first_idx]
            delta_log_y = log_y[last_idx] - log_y[first_idx]
            
            if delta_x > 0 && delta_log_y < 0
                # Valid decay
                neg_inv_b = delta_log_y / delta_x
                log_a = log_y[first_idx] - neg_inv_b * x_filtered[first_idx]
                
                a = exp(log_a) * scale_factor
                b = -1.0 / neg_inv_b
                
                # Safety limits
                b = clamp(b, 1.0, 1000.0)
                
                return b, a
            end
        end
        
        # If all else fails
        return 0.0, scale_factor
    end
end

"""
    fit_t2star_nonlinear(x, echo_times, b0_map, s0_phase; mask=nothing)

Fit T2* and S0 magnitude using nonlinear least squares.
This can be more accurate but slower than the linear fitting method.

Requires Optim.jl package.
"""
function fit_t2star_nonlinear(x, echo_times, b0_map, s0_phase; mask=nothing)
    
    # Get dimensions
    nx, ny, nz, necho = size(x)
    
    # Initialize output maps
    t2star_map = zeros(Float64, nx, ny, nz)
    s0_mag = zeros(Float64, nx, ny, nz)
    
    # Create a default mask if none provided
    if isnothing(mask)
        mask = trues(nx, ny, nz)
    end
    
    # Function to calculate the residual for fitting
    function cost_function(params, te_vals, signal_vals)
        s0 = params[1]
        t2star = params[2]
        
        # Safety check to prevent numerical issues
        if t2star <= 0 || !isfinite(t2star) || !isfinite(s0) || s0 <= 0
            return Inf
        end
        
        # Compute model signal with safety to prevent overflow
        model = similar(signal_vals)
        for i in eachindex(te_vals)
            # Avoid very negative exponents that could cause numeric issues
            if te_vals[i] / t2star > 20.0
                model[i] = 0.0
            else
                model[i] = s0 * exp(-te_vals[i] / t2star)
            end
        end
        
        # Return sum of squared residuals
        return sum((model - signal_vals).^2)
    end
    
    # Prepare for fitting
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                if !mask[i,j,k]
                    continue
                end
                
                try
                    # Extract the signal time course for this voxel
                    signal_timecourse = vec(x[i,j,k,:])
                    
                    # Check for valid signal
                    if all(abs.(signal_timecourse) .< eps())
                        continue  # Skip voxels with no signal
                    end
                    
                    # Extract B0 and S0 phase for this voxel
                    b0_val = b0_map[i,j,k]
                    phase_val = s0_phase[i,j,k]
                    
                    # Handle non-finite B0 or phase values
                    if !isfinite(b0_val) || !isfinite(phase_val)
                        # Fall back to basic fit without phase correction
                        magnitude_data = abs.(signal_timecourse)
                    else
                        # Remove the B0-induced phase evolution and initial phase
                        corrected_signal = signal_timecourse .* exp.(-im * (γ * b0_val .* echo_times .+ phase_val))
                        
                        # Take magnitude for T2* fitting
                        magnitude_data = abs.(corrected_signal)
                    end
                    
                    # Check for valid magnitude data after correction
                    if any(!isfinite.(magnitude_data)) || all(magnitude_data .< eps())
                        continue
                    end
                    
                    # Normalize data for better numerical stability
                    scale_factor = maximum(magnitude_data)
                    if scale_factor <= eps()
                        continue
                    end
                    
                    normalized_data = magnitude_data ./ scale_factor
                    
                    # Initial guess for parameters
                    initial_s0 = 1.0  # Using normalized data
                    
                    # Rough estimate for T2* initial guess using linear fit first
                    # This provides a more stable starting point for nonlinear optimization
                    t2star_init, _ = fit_monoexponential(echo_times, normalized_data)
                    
                    # If linear fit gives unreasonable value, use a default
                    if !isfinite(t2star_init) || t2star_init <= 0.0 || t2star_init > 1.0
                        t2star_init = 0.05  # Default value in seconds
                    end
                    
                    # Safe bounds for parameters
                    lower_bounds = [0.1, 1.0]  # Minimum S0, minimum T2*
                    upper_bounds = [10.0, 1000.0]   # Maximum S0, maximum T2*
                    
                    # Try different optimizers if needed
                    try
                        # First try Nelder-Mead which is more robust but less efficient
                        result = optimize(
                            p -> cost_function(p, echo_times, normalized_data),
                            lower_bounds, 
                            upper_bounds,
                            [initial_s0, t2star_init],
                            Fminbox(NelderMead()),
                            Optim.Options(x_tol=1e-4, f_tol=1e-6, iterations=200, show_trace=false)
                        )
                        
                        # Check if result is reasonable
                        opt_params = Optim.minimizer(result)
                        
                        if !Optim.converged(result) || any(!isfinite.(opt_params)) || opt_params[2] <= 0
                            # Fall back to linear fit
                            t2star_val, s0_mag_val = fit_monoexponential(echo_times, magnitude_data)
                        else
                            # Scale back the S0 value
                            s0_mag_val = opt_params[1] * scale_factor
                            t2star_val = opt_params[2]
                        end
                    catch e
                        # If optimization fails, fall back to linear fit
                        t2star_val, s0_mag_val = fit_monoexponential(echo_times, magnitude_data)
                    end
                    
                    # Final safety check
                    if !isfinite(t2star_val) || t2star_val <= 0 || !isfinite(s0_mag_val) || s0_mag_val < 0
                        t2star_val = 0.0
                        s0_mag_val = 0.0
                    end
                    
                    # Store results
                    t2star_map[i,j,k] = t2star_val
                    s0_mag[i,j,k] = s0_mag_val
                    
                catch e
                    # If anything fails, set default values and continue
                    t2star_map[i,j,k] = 0.0
                    s0_mag[i,j,k] = 0.0
                    # println("Error at voxel ($i,$j,$k): ", e)
                end
            end
        end
    end
    
    return t2star_map, s0_mag
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
    s0_phase = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/init_phase$comb$dcf"))

    println("Data Loaded")

    t2, s0_mag = fit_t2star_nonlinear(x, echo_times, Δb0, s0_phase)

    s0 = s0_mag .* cis.(s0_phase)

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/t2$comb$dcf", ComplexF32.(t2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$comb$dcf", ComplexF32.(s0))
end