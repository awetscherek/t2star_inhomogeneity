function adjoint_operator_impl(plan1, r, r2_d, b0_d, s0_fat_d, s0_water_d, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
    timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, nx, ny, nz, nchan)

    g_r_water_t = Array{ComplexF64}(undef, nx, ny, nz * nchan)
    g_r_fat_t = Array{ComplexF64}(undef, nx, ny, nz * nchan)

    # Initialize sum of gradients (nx,ny,nz,nchan) prior to summing of gradients over coils
    g_s0_fat_total = zeros(ComplexF64, nx,ny,nz,nchan)
    g_s0_water_total = zeros(ComplexF64, nx,ny,nz,nchan)
    g_r2_total = zeros(Float64, nx,ny,nz,nchan)
    g_b0_total = zeros(Float64, nx,ny,nz,nchan)

    start_idx = 1
    for t in ProgressBar(1:num_timepoints)
        t_start = (t - 1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, num_total_timepoints)
        
        t_ms = approximate_time(timepoints, t_start, t_end, :nearest_neighbour)

        # Get the boolean mask for timepoint t (assume selection is 2D with one row per timepoint)
        sel = selection[t_start:t_end, :]
        npoints = sum(sel)

        # Extract the segment of the residual corresponding to timepoint t.
        r_t = r[start_idx:start_idx+npoints-1, :]
        fat_modulation_t = !isnothing(fat_modulation) ? view(fat_modulation,start_idx:start_idx+npoints-1) : 1.0

        dcf_t = use_dcf ? view(dcf_d, start_idx:start_idx+npoints-1) : 1.0

        kx_d_t = collect(kx_d[t_start:t_end, :][sel])
        ky_d_t = collect(ky_d[t_start:t_end, :][sel])

        finufft_setpts!(plan1, kx_d_t, ky_d_t)

        finufft_exec!(plan1, r_t .* dcf_t .* conj.(fat_modulation_t), g_r_fat_t)
        finufft_exec!(plan1, r_t .* dcf_t , g_r_water_t)

        if combine_coils
            g_r_fat_result_t = reshape(g_r_fat_t, size(c_d)) .* conj(c_d)
            g_r_water_result_t = reshape(g_r_water_t, size(c_d)) .* conj(c_d)
        else
            g_r_fat_result_t = reshape(g_r_fat_t, size(r2_d))
            g_r_water_result_t = reshape(g_r_water_t, size(r2_d))
        end

        conj_s0_fat = conj.(s0_fat_d)
        conj_s0_water = conj.(s0_water_d)

        conj_exp_term = conj.(exp.(t_ms .* (im .* γ .* b0_d .- r2_d)))

        g_r2_total .+= real.(
            (- conj_s0_water .* t_ms .* conj_exp_term) .* g_r_water_result_t
            .+ (- conj_s0_fat .* t_ms .* conj_exp_term) .* g_r_fat_result_t
            )

        g_b0_total .+= real.(
            (- im .* γ .* t_ms .* conj_s0_water .* conj_exp_term) .* g_r_water_result_t
            .+ (- im .* γ .* t_ms .* conj_s0_fat .* conj_exp_term) .* g_r_fat_result_t
        )
        
        g_s0_fat_total .+= conj_exp_term .* g_r_fat_result_t
        g_s0_water_total .+= conj_exp_term .* g_r_water_result_t

        #TODO: Maybe put the sum of gradients in for loop instead of at end

        start_idx += npoints
    end

    if combine_coils
        g_r2_total = dropdims(sum(g_r2_total, dims=4), dims=4)
        g_b0_total = dropdims(sum(g_b0_total, dims=4), dims=4)
        g_s0_fat_total = dropdims(sum(g_s0_fat_total, dims=4), dims=4)
        g_s0_water_total = dropdims(sum(g_s0_water_total, dims=4), dims=4)
    end

    return g_r2_total, g_b0_total, g_s0_fat_total, g_s0_water_total
end

function adjoint_operator_impl(plan1, r, r2_d, b0_d, s0_d, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
    timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, nx, ny, nz, nchan)

    g_r_t = Array{ComplexF64}(undef, nx, ny, nz * nchan)

    # Initialize sum of gradients (nx,ny,nz,nchan) prior to summing of gradients over coils
    g_s0_total = zeros(ComplexF64, nx,ny,nz,nchan)
    g_r2_total = zeros(Float64, nx,ny,nz,nchan)
    g_b0_total = zeros(Float64, nx,ny,nz,nchan)

    start_idx = 1
    for t in ProgressBar(1:num_timepoints)
        t_start = (t - 1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, num_total_timepoints)
        
        t_ms = approximate_time(timepoints, t_start, t_end, :nearest_neighbour)

        # Get the boolean mask for timepoint t (assume selection is 2D with one row per timepoint)
        sel = selection[t_start:t_end, :]
        npoints = sum(sel)

        # Extract the segment of the residual corresponding to timepoint t.
        r_t = r[start_idx:start_idx+npoints-1, :]

        dcf_t = use_dcf ? view(dcf_d, start_idx:start_idx+npoints-1) : 1.0

        kx_d_t = collect(kx_d[t_start:t_end, :][sel])
        ky_d_t = collect(ky_d[t_start:t_end, :][sel])

        finufft_setpts!(plan1, kx_d_t, ky_d_t)

        finufft_exec!(plan1, r_t .* dcf_t, g_r_t)

        if combine_coils
            g_r_result_t = reshape(g_r_t, size(c_d)) .* conj(c_d)
        else
            g_r_result_t = reshape(g_r_t, size(r2_d))
        end

        conj_s0 = conj.(s0_d)
        conj_exp_term = conj.(exp.(t_ms .* (im .* γ .* b0_d .- r2_d)))

        g_r2_total .+= real.((- conj_s0 .* t_ms .* conj_exp_term) .* g_r_result_t)
        g_b0_total .+= real.((- im .* γ .* t_ms .* conj_s0 .* conj_exp_term) .* g_r_result_t)
        g_s0_total .+= conj_exp_term .* g_r_result_t

        start_idx += npoints
    end

    if combine_coils
        g_r2_total = dropdims(sum(g_r2_total, dims=4), dims=4)
        g_b0_total = dropdims(sum(g_b0_total, dims=4), dims=4)
        g_s0_total = dropdims(sum(g_s0_total, dims=4), dims=4)
    end

    return g_r2_total, g_b0_total, g_s0_total
end

function adjoint_operator_impl(plan1, r, e_d, s0_d, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
    timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, nx, ny, nz, nchan)

    g_r_t = Array{ComplexF64}(undef, nx, ny, nz * nchan)

    # Initialize sum of gradients (nx,ny,nz,nchan) prior to summing of gradients over coils
    g_s0_total = zeros(ComplexF64, nx, ny, nz, nchan)
    g_e_total = zeros(ComplexF64, nx, ny, nz, nchan)

    start_idx = 1
    for t in ProgressBar(1:num_timepoints)
        t_start = (t - 1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, num_total_timepoints)

        t_ms = approximate_time(timepoints, t_start, t_end, :nearest_neighbour)

        # Get the boolean mask for timepoint t (assume selection is 2D with one row per timepoint)
        sel = selection[t_start:t_end, :]
        npoints = sum(sel)

        # Extract the segment of the residual corresponding to timepoint t.
        r_t = r[start_idx:start_idx+npoints-1, :]
        # fat_modulation_t = fat_modulation[start_idx:start_idx+npoints-1]

        dcf_t = use_dcf ? view(dcf_d, start_idx:start_idx+npoints-1) : 1.0

        kx_d_t = collect(kx_d[t_start:t_end, :][sel])
        ky_d_t = collect(ky_d[t_start:t_end, :][sel])

        finufft_setpts!(plan1, kx_d_t, ky_d_t)

        finufft_exec!(plan1, r_t .* dcf_t, g_r_t)

        if combine_coils
            g_r_result_t = reshape(g_r_t, size(c_d)) .* conj(c_d)
        else
            g_r_result_t = reshape(g_r_t, size(e_d))
        end

        conj_s0 = conj.(s0_d)
        conj_exp_term = conj.(exp.(-t_ms .* e_d))

        g_e_total .+= (-conj_s0 .* t_ms .* conj_exp_term) .* g_r_result_t
        g_s0_total .+= conj_exp_term .* g_r_result_t

        #TODO: Maybe put the sum of gradients in for loop instead of at end

        start_idx += npoints
    end

    if combine_coils
        g_e_total = dropdims(sum(g_e_total, dims=4), dims=4)
        g_s0_total = dropdims(sum(g_s0_total, dims=4), dims=4)
    end

    return g_e_total, g_s0_total
end