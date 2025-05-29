function preprocess_data(config, raw, combine_coils, sens, kx, ky, timepoint_window_size, use_dcf, fat_modulation, use_synthetic=false)
    if !use_synthetic
        @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."
    end

    if length(size(kx)) == 3
        kx = kx[:, :, :, :]
    end

    if length(size(ky)) == 3
        ky = ky[:, :, :, :]
    end

    nkx, necho, nky,_= size(kx)

    @assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

    if use_synthetic
        c_d = combine_coils ? ones(Float64, nx, ny, nz, config["nchan"]) : [1.0]
    else
        c_d = combine_coils ? sens : [1.0]
    end
    

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [3 1 2 4]) * config["FOVx"] / nx * 2 * pi, nky, :)
    ky_d = reshape(permutedims(ky, [3 1 2 4]) * config["FOVy"] / ny * 2 * pi, nky, :)

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi

    dcf_y = use_dcf ? reshape(sqrt.(dcf), 1, size(dcf, 1), 1, 1, 1) : dcf

    dcf_d = use_dcf ? reshape(repeat(sqrt.(dcf), outer=(size(ky, 2), size(ky, 3))),nky,:)[selection] : 1.0

    if use_synthetic
        y_d = raw #generated y_d data passed in (multiplied by dcf as required in load_synthetic_data)
    else
        y_d = reshape(ComplexF64.(permutedims(raw, [5 1 3 4 2])) .* dcf_y, nky, :, nz * config["nchan"])[selection, :]
    end

    num_total_timepoints = config["necho"] * nkx
    num_timepoints = ceil(Int, num_total_timepoints / timepoint_window_size)

    # Reshape fat modulation so it can be easily multiplied in k-space
    if !isnothing(fat_modulation)
        fat_modulation = repeat(reshape(vec(fat_modulation),1,:), nky, 1)[selection]
    end

    return y_d, kx_d, ky_d, dcf_d, c_d, selection, num_timepoints, num_total_timepoints, fat_modulation
end