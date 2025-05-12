function forward_operator_impl(plan2, e_d, s0_fat_d, s0_water_d, num_timepoints, num_total_timepoints, kx_d, ky_d,
    c_d, timepoints, selection, timepoint_window_size, fat_modulation = nothing)
    if !isnothing(fat_modulation)
        y_water = _forward_operator_impl(plan2, e_d, s0_water_d, num_timepoints, num_total_timepoints, kx_d, ky_d,
        c_d, timepoints, selection, timepoint_window_size)
        y_fat = _forward_operator_impl(plan2, e_d, s0_fat_d, num_timepoints, num_total_timepoints, kx_d, ky_d,
        c_d, timepoints, selection, timepoint_window_size)
        return y_water .+ fat_modulation .* y_fat
    else
        # Not utilising Fat Modulation, implicitly assume that everything is Water
        return _forward_operator_impl(plan2, e_d, s0_water_d, num_timepoints, num_total_timepoints, kx_d, ky_d,
        c_d, timepoints, selection, timepoint_window_size)
    end
end

function _forward_operator_impl(plan2, e_d, s0_d, num_timepoints, num_total_timepoints, kx_d, ky_d,
    c_d, timepoints, selection, timepoint_window_size)
    y_list = Vector{Array{ComplexF64}}(undef, num_timepoints)
    for t in ProgressBar(1:num_timepoints)
        t_start = (t - 1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, num_total_timepoints)
        
        t_ms = approximate_time(timepoints, t_start, t_end, :nearest_neighbour)

        sel = selection[t_start:t_end, :]
        kx_d_t = collect(kx_d[t_start:t_end, :][sel])
        ky_d_t = collect(ky_d[t_start:t_end, :][sel])

        finufft_setpts!(plan2, kx_d_t, ky_d_t)

        w_d_t = s0_d .* exp.(-t_ms .* e_d)

        y_t = finufft_exec(plan2, w_d_t .* c_d)

        y_list[t] = y_t
    end
    y = vcat(y_list...)
    return y
end