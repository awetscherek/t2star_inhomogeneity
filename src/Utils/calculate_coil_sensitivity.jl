function calculate_coil_sensitivity(config, kx, ky, raw)
    @info "Using Combined coils"

    if !isfile("/mnt/f/Dominic/Data/coil_sens/Real/sens.cfl")
        @info "No coil sensitivies found - creating coil sensitivity estimation"

        # low resolution reconstruction of echo 1 for coil sensitivity estimation:
        x = image_recon_2d(config,
            @view(kx[:, 1, :]),
            @view(ky[:, 1, :]),
            @view(raw[:, :, 1, :, :]),
            [64, 64]
        )

        ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/coil_sens/Real/lowres_img", ComplexF32.(x))

        # run external tool to estimate coil sensitivities (and interpolate to full image resolution):
        run(`../../bart-0.9.00/bart fft -u 7 /mnt/f/Dominic/Data/coil_sens/Real/lowres_img /mnt/f/Dominic/Data/coil_sens/Real/lowres_ksp`)
        run(`../../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz /mnt/f/Dominic/Data/coil_sens/Real/lowres_ksp /mnt/f/Dominic/Data/coil_sens/Real/ksp_zerop`)
        run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 /mnt/f/Dominic/Data/coil_sens/Real/ksp_zerop /mnt/f/Dominic/Data/coil_sens/Real/sens`)
    end
    # load coil sensitivities into Julia 
    return ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/coil_sens/Real/sens")
end

function calculate_synthetic_coil_sensitivity(config, eval_no, kx, ky, y_d)
    return ones(Float64, nx, ny, nz, config["nchan"])
    @info "Using Synthetic Combined Coils"

    if !isfile("/mnt/f/Dominic/Data/coil_sens/Synthetic/$(eval_no)_sens.cfl")
        @info "No coil sensitivies found for synthetic data ($eval_no) - creating coil sensitivity estimation for synthetic data"

        # low resolution reconstruction of echo 1 for coil sensitivity estimation:
        x = image_recon_synthetic_2d(config, 
            @view(kx[:, 1, :]),
            @view(ky[:, 1, :]),
            y_d,
            )

        ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/coil_sens/Synthetic/coil_img", ComplexF32.(x))

        # run external tool to estimate coil sensitivities (and interpolate to full image resolution):
        run(`../../bart-0.9.00/bart fft -u 7 /mnt/f/Dominic/Data/coil_sens/Synthetic/coil_img /mnt/f/Dominic/Data/coil_sens/Synthetic/lowres_ksp`)
        # run(`../../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz /mnt/f/Dominic/Data/coil_sens/Synthetic/lowres_ksp /mnt/f/Dominic/Data/coil_sens/Synthetic/ksp_zerop`)
        run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 /mnt/f/Dominic/Data/coil_sens/Synthetic/lowres_ksp /mnt/f/Dominic/Data/coil_sens/Synthetic/$(eval_no)_sens`)
    end
    # load coil sensitivities into Julia
    return ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/coil_sens/Synthetic/$(eval_no)_sens.cfl")
end

function calculate_synthetic_coil_sensitivity(config)
    return ones(Float64, nx, ny, nz, config["nchan"])
end