using FINUFFT

function recon_2d_t2star_map(config, kx, ky, raw, time_since_last_rf, dims; # keyword arguments: 
    combine_coils = false,      # whether to use coil sensitivities
    sens = nothing,             # coil sensitivities ...
    use_dcf = false,            # whether to use pre-conditioner
    tol = 1e-9,                 # tolerance for FINUFFT
    niter = use_dcf ? 10 : 100, # number of gradient descent iterations
    max_ls = 100,               # number of line search iterations
    alpha0 = 1.0,               # initial step size for line search
    beta = 0.5)                 # factor to decrease step size

    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    #kx and ky should be of shape
    #(269,8,536)
    # nky => necho => nkx
    nx, ny = dims;
    nz = size(raw, 4) # this assumes only data from one echo is passed to the function, but several slices, so raw should be a 4D array

    nkx, _, nky,_ = size(kx)

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

    c_d = combine_coils ? sens : [1.0]; # this shouldn't make a copy of sens

    # println("kx and ky shape")
    # println(size(kx))
    # println(size(ky))

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [3 1 2 4]) * config["FOVx"] / nx * 2 * pi, :, nky);
    ky_d = reshape(permutedims(ky, [3 1 2 4]) * config["FOVy"] / ny * 2 * pi, :, nky);

    # println("kx_d and ky_d shape")
    # println(size(kx_d))
    # println(size(ky_d))

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi;

    # println("selection shape")
    # println(size(selection))

    # this is where the coil images will be stored - note that we still will reconstruct several slices:
    t2_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
    s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

    # println("size of t2_d")
    # println(size(t2_d))

    # this is the raw data from which we want to reconstruct the coil images
    #(timepoints, ky, nz * nchan)

    # println("Raw shape")
    # println(size(raw))

    y_d = reshape(ComplexF64.(permutedims(raw,[1 3 5 4 2])) .* sqrt.(dcf), nkx * config["necho"], :, nz * config["nchan"])[selection, :];

    println("y_d shape")
    println(size(y_d))

    dcf_d = use_dcf ? reshape(repeat(sqrt.(dcf), outer = (1, size(ky, 2))), :)[selection] : 1.0;
    
    #Precision of approximation of timepoints
    # 1 - No approximation (NUFFT for every time point)
    # nkx (536) - Echo time of each assumed to be the timepoint
    timepoint_window_size = 536

    @assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

    total_timepoints = config["necho"] * nkx
    timepoints = ceil(Int, total_timepoints / timepoint_window_size)

    # initial guess of T2 and S0:
    t2_d .= 50.0;
    s0_d .= 1.0;

    #Benchmark of forward operator using  T2 and S0 mappings from Intermediate Generation
    # if combine_coils
    #     t2_d = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/t2star_2d"))
    #     s0_d = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/s0_2d"))
    # else
    #     t2_d = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/t2star_2d_no_combine_coils"))
    #     s0_d = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/s0_2d_no_combine_coils"))
    # end

    time_since_last_rf = vec(time_since_last_rf)

    # allocate some arrays:
    u_t2 = Array{ComplexF64}(undef, size(y_d));
    u_s0 = Array{ComplexF64}(undef, size(y_d));

    tmp = Array{ComplexF64}(undef, size(y_d));

    # intermediate results, needed to combine gradient across coils...
    # g_r_tmp = Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);
    # g_r = combine_coils ? Array{ComplexF64}(undef, size(t2_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);
    # g_r_t
    g_t2 = combine_coils ? Array{ComplexF64}(undef, size(t2_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);
    g_s0 = combine_coils ? Array{ComplexF64}(undef, size(t2_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);

    println("For the plan1 nufft")
    println("kx_d")
    println(size(kx_d[selection]))
    println("ky_d")
    println(size(ky_d[selection]))

    # plan NUFFTs:
    plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
    # finufft_setpts!(plan1, kx_d[selection], ky_d[selection])

    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    # plan3 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)
    # finufft_setpts!(plan3, kx_d[selection], ky_d[selection])

    # r = Array{ComplexF64}(undef,size(y_d));
    println("size of r should be the same as y_d which is")
    println(size(y_d))
    y = forward_operator(plan2, t2_d, s0_d, timepoints, total_timepoints, kx_d, ky_d, c_d, time_since_last_rf, selection, timepoint_window_size)
    y .*= dcf_d;
    r = y .- y_d;

    obj = real(r[:]' * r[:]) / 2.0; # objective function
    alpha_s0 = alpha0 * beta
    alpha_t2 = alpha0 * beta

    initial_obj = "Initial obj = $obj"
    @info initial_obj

    open("output.txt", "a") do f
        println(f, string(initial_obj))
    end

    for it = 1:niter
        alpha_s0 /= beta
        alpha_t2 /= beta
        obj0 = obj
        
        # g_s0, g_t2 = jacobian_operator(plan1, g_r, g_r_tmp, r, t2_d, s0_d, dcf_d, combine_coils, c_d, timepoints, time_since_last_rf)
        g_s0, g_t2 = jacobian_operator(plan1, r, t2_d, s0_d, dcf_d, combine_coils, c_d, timepoints, total_timepoints, time_since_last_rf, kx_d ,ky_d, selection, use_dcf, timepoint_window_size)

        # finufft_exec!(plan3, g_s0 .* c_d, u_s0)
        # finufft_exec!(plan3, g_t2 .* c_d, u_t2)
        println("forward operator on gradient")
        u_t2 = forward_operator(plan2, g_t2, s0_d, timepoints, total_timepoints, kx_d, ky_d, c_d, time_since_last_rf, selection, timepoint_window_size)
    

        # println("size of u_s0")
        # println(size(u_s0))
        println("size of u_t2")
        println(size(u_t2))

        # u_s0 .*= dcf_d;
        u_t2 .*= dcf_d;

        # println("Performing Line search for S0")
        # #Line search for s0
        # for _ = 1:max_ls 
        #     alpha_s0 *= beta
        #     tmp .= r .- alpha_s0 * u_s0;
        #     obj = real(tmp[:]' * tmp[:]) / 2.0
        #     obj < obj0 && break
        # end

        println("Performing Line search to T2")
        #Line search for t2
        for it = 1:max_ls 
            println("Iter: $it")
            alpha_t2 *= beta
            tmp .= r .- alpha_t2 * u_t2;
            obj = real(tmp[:]' * tmp[:]) / 2.0
            obj < obj0 && break
        end

        # @info "alpha_t2 = $alpha_t2, alpha_s0 = $alpha_s0"
        @info "alpha_t2 = $alpha_t2"

        # Gradient Descent Updates
        t2_d .-= alpha_t2 * reshape(g_t2, size(t2_d));
        # s0_d .-= alpha_s0 * reshape(g_s0, size(s0_d));
        s0_d .-= alpha_t2 * reshape(g_s0, size(s0_d));

        y = forward_operator(plan2, t2_d, s0_d, timepoints, total_timepoints, kx_d, ky_d, c_d, time_since_last_rf, selection, timepoint_window_size)
        y .*= dcf_d;
        r = y .- y_d;

        obj = real(r[:]' * r[:]) / 2.0

        # info="it = $it, alpha_t2 = $alpha_t2, alpha_s0 = $alpha_s0 obj = $obj"
        info="it = $it, alpha_t2 = $alpha_t2, obj = $obj"
        @info info
        open("output.txt", "a") do f
            println(f, string(info))
        end
    end

    finufft_destroy!(plan1)
    finufft_destroy!(plan2)
    # finufft_destroy!(plan3)

    # collect results from GPU & return: 
    t2_d
end

function forward_operator(plan2, t2_d, s0_d, timepoints, total_timepoints, kx_d, ky_d, c_d, time_since_last_rf, selection, timepoint_window_size)
    y_list = Vector{Array{ComplexF64}}(undef, timepoints)
    for t in 1:timepoints
        @info "t=$t"
        t_start = (t-1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, total_timepoints)
        approx_t = round(Int, (t_start + t_end) / 2)

        t_ms = time_since_last_rf[approx_t]
 
        kx_d_t = collect(kx_d[t_start:t_end,:][selection[t_start:t_end,:]])
        ky_d_t = collect(ky_d[t_start:t_end,:][selection[t_start:t_end,:]])

        # println("kx_d_t")
        # println(size(kx_d_t))
        # println("ky_d_t")
        # println(size(ky_d_t))

        finufft_setpts!(plan2, kx_d_t, ky_d_t)

        # calculate the residual
        w_d_t = s0_d .* exp.(- t_ms ./ t2_d)

        y_t = finufft_exec(plan2, w_d_t .* c_d)

        println("size of y_t")
        println(size(y_t))
        
        y_list[t] = y_t
    end
    y = vcat(y_list...)
    println("size of y")
    println(size(y))

    # z = zeros(ComplexF64, 1141624,256)
    # return z
    return y
end

# function jacobian_operator(plan1, g_r, g_r_tmp, r, t2_d, s0_d, dcf_d, combine_coils, c_d, timepoints, time_since_last_rf)

#     if combine_coils
#         finufft_exec!(plan1, r .* dcf_d, g_r_tmp)
#         g_r .= sum(reshape(g_r_tmp, size(c_d)) .* conj(c_d), dims=4); 
#     else
#         finufft_exec!(plan1, r .* dcf_d, g_r)
#     end

#     # for line search, we can use that (NU)FFT is a linear operator, so do NUFFT only once:
#     #w_d = s0 * e ^ (-t / T2*)

#     println("shape of g_r")
#     println(size(g_r))
#     g_r_reshaped = reshape(g_r, size(t2_d))
#     println("shape of g_r after reshape")
#     println(size(g_r_reshaped))

#     g_t2_tmp = Array{ComplexF64}(undef, size(t2_d))
#     g_s0_tmp = Array{ComplexF64}(undef, size(t2_d))

#     g_t2_tmp .= 0
#     g_s0_tmp .= 0 

#     for t in 1:timepoints
#         @info "Jacobian Operator t=$t"
#         t_ms = time_since_last_rf[t]
#         #maybe should be +=?
#         g_s0_tmp .+= (g_r_reshaped .* exp.(-t_ms ./ t2_d))
#         g_t2_tmp .+= (g_r_reshaped .* s0_d .* exp.(-t_ms ./ t2_d) .* (t_ms ./ t2_d.^2))
#     end

#     # g_s0_tmp .*= -1

#     return reshape(g_s0_tmp, size(g_r)), reshape(g_t2_tmp, size(g_r))
# end

function jacobian_operator(plan1, r, t2_d, s0_d, dcf_d, combine_coils, c_d, 
    timepoints, total_timepoints, time_since_last_rf, kx_d, ky_d, selection, use_dcf, timepoint_window_size)
    # Initialize gradients (same size as the image)
    g_s0_total = zeros(ComplexF64, size(t2_d))
    g_t2_total = zeros(ComplexF64, size(t2_d))

    # Pointer into the concatenated residual vector r.
    start_idx = 1
    for t in 1:timepoints
        @info "Jacobian Operator t=$t"

        t_start = (t-1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, total_timepoints)
        approx_t = round(Int, (t_start + t_end) / 2)

        t_ms = time_since_last_rf[approx_t]

        # Get the boolean mask for timepoint t (assume selection is 2D with one row per timepoint)
        sel = selection[t_start:t_end, :]
        npoints = sum(sel)

        # Extract the segment of the residual corresponding to timepoint t.
        r_t = r[start_idx:start_idx+npoints-1]
        dcf_t = use_dcf ? dcf_d[start_idx:start_idx+npoints-1] : 1.0

        # Extract the corresponding kx and ky for this timepoint.
        kx_d_t = collect(kx_d[t_start:t_end, :][sel])
        ky_d_t = collect(ky_d[t_start:t_end, :][sel])

        # println("size of kx_d_t")
        # println(size(kx_d_t))
        # println("size of ky_d_t")
        # println(size(ky_d_t))

        # Set the NUFFT points for plan1 for this timepoint.
        finufft_setpts!(plan1, kx_d_t, ky_d_t)

        # Allocate a temporary array for the adjoint output.
        # (Assuming the NUFFT returns an image of size equal to t2_d)
        # g_r_t = similar(t2_d)
        #should be nkx, nky, nkz*nchan

        # Compute the adjoint NUFFT for the t-th timepoint.
        g_r_t = finufft_exec(plan1, r_t .* dcf_t)

        # If coil combination is enabled, you would combine using the coil sensitivities.
        # Here, for simplicity, we assume that either the combination has been handled
        # already or that c_d == 1.
        if combine_coils
            # For example, if c_d is provided per coil, combine as:
            # g_r_t = sum(g_r_t .* conj(c_d), dims=coil_dimension)
            g_r_t = sum(reshape(g_r_t, size(c_d)) .* conj(c_d), dims=4);
        end

        # Weight the adjoint result with the appropriate exponential factors.
        g_r_t = dropdims(g_r_t; dims=4)

        dt_dt2 = - exp.(-t_ms ./ t2_d)
        g_s0_t = g_r_t .* dt_dt2
        g_s0_total .+= g_s0_t
        g_t2_total .+= (g_s0_t .* s0_d .* (t_ms ./ t2_d.^2))

        # Update start index for the next timepoint's residual segment.
        start_idx += npoints
    end
    return g_s0_total, g_t2_total
end

