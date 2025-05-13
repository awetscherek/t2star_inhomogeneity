function load_and_process_data(combine_coils :: Bool, use_fat_modulation :: Bool)
    config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true);

    @assert size(noise) == (19832, 8)   # noise measurement could be used for pre-whitening
    @assert size(raw) == (536, 8, 8, 32, 269)
    # acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

    # perform FFT across slices:
    I = sortperm(-kz[1, 1, :, 1]); # looks like we need to flip the sign to make this work with standard FFT implementations ... we use ifft to account for this minus ...
    v = kz[1, 1, I, 1];
    @assert all(v ./ v[18] .≈ -16:15) # v[18] is the first positive kz coordinate, so basically v[18] = delta kz = 1/FOV

    # this is working on the raw data, so it should only be executed once:
    raw = raw[:, :, :, I, :];   # sorting by kz ...
    raw = ifftshift(raw, 4); # since even number of partitions, fftshift and ifftshift will do the same thing ...
    ifft!(raw, 4);
    raw .*= sqrt(size(raw, 4)); # maintaining the norm ...
    raw = fftshift(raw, 4);

    # we'll use the kx and ky trajectory from the k-space centre plane:
    kx = kx[:, :, 1, :]
    ky = ky[:, :, 1, :]
    kz = nothing

    if combine_coils
        sens = calculate_coil_sensitivity(config, kx, ky, raw)
    end
    #######################################################################################################################

    if use_fat_modulation
        @info "Using Fat Modulation"
        fat_modulation = calculate_fat_modulation(time_since_last_rf)
    end

    timepoints = vec(time_since_last_rf)

    #Convert ms to sec
    timepoints .*= 1e-3

    return raw, kx, ky, kz, config, sens, timepoints, use_fat_modulation ? fat_modulation : nothing
end