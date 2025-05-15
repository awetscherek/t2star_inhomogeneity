function calculate_coil_sensitivity(config, kx, ky, raw)
    @info "Using Combined coils"

    if !isfile("data/coil_sens/sens.cfl")
        @info "No coil sensitivies found - creating coil sensitivity estimation"

        # low resolution reconstruction of echo 1 for coil sensitivity estimation:
        x = image_recon_2d(config,
            @view(kx[:, 1, :]),
            @view(ky[:, 1, :]),
            @view(raw[:, :, 1, :, :]),
            [64, 64]
        )

        ReadWriteCFL.writecfl("data/coil_sens/lowres_img", ComplexF32.(x))

        # run external tool to estimate coil sensitivities (and interpolate to full image resolution):
        run(`../../bart-0.9.00/bart fft -u 7 data/coil_sens/lowres_img data/coil_sens/lowres_ksp`)
        run(`../../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz data/coil_sens/lowres_ksp data/coil_sens/ksp_zerop`)
        run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 data/coil_sens/ksp_zerop data/coil_sens/sens`)
    end
    # load coil sensitivities into Julia
    return ReadWriteCFL.readcfl("data/coil_sens/sens")
end

function calculate_synthetic_coil_sensitivity(sens)
    @info "Using Combined coils"
    sens .= 1
end