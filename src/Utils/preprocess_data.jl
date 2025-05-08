function preprocess_data(config, raw, combine_coils, sens, kx, ky, timepoint_window_size, use_dcf, fat_modulation)
    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    #kx and ky should be of shape
    #(269,8,536)
    # nky => necho => nkx
    nkx, _, nky, _ = size(kx)

    @assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

    c_d = combine_coils ? sens : [1.0] # this shouldn't make a copy of sens

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [2 1 3 4]) * config["FOVx"] / nx * 2 * pi, :, nky)
    ky_d = reshape(permutedims(ky, [2 1 3 4]) * config["FOVy"] / ny * 2 * pi, :, nky)

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi

    dcf_y = use_dcf ? reshape(sqrt.(dcf), 1, size(dcf, 1), 1, 1, 1) : dcf
    dcf_d = use_dcf ? repeat(sqrt.(dcf), outer=(size(ky, 2), size(ky, 3)))[selection] : 1.0

    y_d = reshape(ComplexF64.(permutedims(raw, [3 1 5 4 2])) .* dcf_y, config["necho"] * nkx, :, nz * config["nchan"])[selection, :]

    num_total_timepoints = config["necho"] * nkx
    num_timepoints = ceil(Int, num_total_timepoints / timepoint_window_size)

    # Reshape fat modulation so it can be easily multiplied in k-space
    if !isnothing(fat_modulation)
        fat_modulation = repeat(vec(fat_modulation), 1, nky)[selection]
    end

    return y_d, kx_d, ky_d, dcf_d, c_d, selection, num_timepoints, num_total_timepoints, fat_modulation
end