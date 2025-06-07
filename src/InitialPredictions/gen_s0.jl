function fit_t2star(x, echo_times; mask=nothing)
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
                magnitude_data = abs.(x[i,j,k,:])

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

    b = min(b, 200)
    
    return b, a  # Return T2* and |S₀|
end