using DqT2

# Configure Settings
combine_coils = true
use_dcf = true

raw, kx, ky, kz, config, sens, timepoints, _ = load_and_process_data(combine_coils, false)

#######################################################################################################################

# full-scale reconstruction (can loop over echoes):
x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

for (ie, xe) in zip(1:config["necho"], eachslice(x, dims=length(size(x))))

    xe .= image_recon_3d(config, 
        @view(kx[:, ie, :, :]),
        @view(ky[:, ie, :, :]),
        @view(kz[:, ie, :, :]),
        @view(raw[:, :, ie, :, :]),
        [nx, ny, nz],
        combine_coils = combine_coils,
        sens = combine_coils ? sens : nothing,
        use_dcf = use_dcf, # for some reason this seems to introduce artifacts into the image, so it might be required to add some regularisation ...
    );

end

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Recon/3d/x$comb$dcf", ComplexF32.(x))
