function apply_forward_op(t2,b0,water,fat,config, kx, ky, timepoints, dims;
                            combine_coils=false,
                            sens=nothing,
                            use_dcf=false,
                            timepoint_window_size=536,
                            fat_modulation=nothing,
                            tol=1e-9)

    (
        _,
        kx_d,
        ky_d,
        dcf_d,
        c_d,
        selection,
        num_timepoints,
        num_total_timepoints,
        fat_modulation
    ) = preprocess_data(
        config,
        nothing,
        combine_coils,
        sens,
        kx,
        ky,
        timepoint_window_size,
        true, #use_dcf
        fat_modulation,
        true
        )

    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    #We return dcf for dcf-weighted rmse in evaluation, but multiply the ksp by dcf depending on the variable
    if use_dcf
        dcf = dcf_d
    else
        dcf = dcf_d
        dcf_d = 1.0
    end 


    #------------------------------------------------------
    #Setup test functions

    # Initialise Operators with implicit values
    function forward_operator(e,fat,water)
        return forward_operator_impl(plan2, e, fat,water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation) .* dcf_d, dcf
    end

    function forward_operator(e,s0)
        return forward_operator_impl(plan2, e, nothing, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation) .* dcf_d, dcf
    end

    r2 = 1 ./ Float64.(t2)
    im = -γ .* Float64.(b0)
    e = complex.(r2, im)

    if isnothing(fat_modulation)
        return forward_operator(e, water)
    else
        return forward_operator(e, fat, water)
    end
end