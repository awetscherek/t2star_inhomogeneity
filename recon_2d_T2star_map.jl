using FINUFFT

function recon_2d_t2star_map(config, kx, ky, raw, time_since_last_rf, dims; # keyword arguments: 
    combine_coils = false,      # whether to use coil sensitivities
    sens = nothing,             # coil sensitivities ...
    use_dcf = false,            # whether to use pre-conditioner
    tol = 1e-9,                 # tolerance for FINUFFT
    niter = use_dcf ? 10 : 100, # number of gradient descent iterations
    max_ls = 100,               # number of line search iterations
    alpha0 = 1.0,               # initial step size for line search
    beta = 0.9)                 # factor to decrease step size

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

    #Do i need to change sens?
    print(size(c_d))

    println("kx and ky shape")
    println(size(kx))
    println(size(ky))

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [3 1 2 4]) * config["FOVx"] / nx * 2 * pi, :, nky);
    ky_d = reshape(permutedims(ky, [3 1 2 4]) * config["FOVy"] / ny * 2 * pi, :, nky);

    println("kx_d and ky_d shape")
    println(size(kx_d))
    println(size(ky_d))

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi;

    println("selection shape")
    println(size(selection))

    # this is where the coil images will be stored - note that we still will reconstruct several slices:
    t2_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
    s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

    # this is the raw data from which we want to reconstruct the coil images
    #(timepoints, ky, nz * nchan)

    println("Raw shape")
    println(size(raw))

    y_d = reshape(ComplexF64.(permutedims(raw,[1 3 5 4 2])) .* sqrt.(dcf), nkx * config["necho"], :, nz * config["nchan"]);

    println("y_d shape")
    println(size(y_d))

    dcf_d = use_dcf ? reshape(repeat(sqrt.(dcf), outer = (1, size(ky, 2))), :)[selection] : 1.0;

    timepoints = config["necho"] * nkx

    # initial guess:
    t2_d .= 50.0;
    s0_d .= 500.0;

    time_since_last_rf = vec(time_since_last_rf)

    for t = 1:timepoints

        y_d_t = y_d[t,:,:][selection[t,:],:]
        kx_d_t = kx_d[t,:][selection[t,:]]
        ky_d_t = ky_d[t,:][selection[t,:]]

        # allocate some arrays:
        g_t2 = combine_coils ? Array{ComplexF64}(undef, size(t2_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);
        g_s0 = combine_coils ? Array{ComplexF64}(undef, size(t2_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);

        u_t2 = Array{ComplexF64}(undef, size(y_d_t));
        u_s0 = Array{ComplexF64}(undef, size(y_d_t));

        tmp_t2 = Array{ComplexF64}(undef, size(y_d_t));
        tmp_s0 = Array{ComplexF64}(undef, size(y_d_t));

        # intermediate results, needed to combine gradient across coils...
        g_r_tmp = Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);
        g_r = combine_coils ? Array{ComplexF64}(undef, size(t2_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);

        # println("kx_d_t dimensions")
        # println(size(kx_d_t))
        # println("ky_d_t dimensions")
        # println(size(ky_d_t))

        r_t = Array{ComplexF64}(undef, size(kx_d_t,1), nz * config["nchan"]);

        # println("size of r_t")
        # println(size(r_t))

        t_ms = time_since_last_rf[t]

        @info "t = $t, t_ms = $t_ms"

        # plan NUFFTs:
        plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)
        finufft_setpts!(plan2, kx_d_t, ky_d_t)
        plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
        finufft_setpts!(plan1, kx_d_t, ky_d_t)

        # calculate the residual
        
        w_d_t = s0_d .* exp.(t_ms ./ t2_d)

        # println("size of w_d_t")
        # println(size(w_d_t))

        finufft_exec!(plan2, w_d_t .* c_d, r_t)
        r_t .*= dcf_d;
        r_t .-= y_d_t;

        # obj = real(r_t[:]' * r_t[:]) / 2.0; # objective function
        obj = real(sum(abs2, r_t)) / 2.0
        alpha_t2 = alpha0 * beta
        alpha_s0 = alpha0 * beta
        @info "obj0 = $obj"
        for it = 1:10
            alpha_t2 /= beta # otherwise step size alpha decreases too fast ... 
            alpha_s0 /= beta
            obj0 = obj
            if combine_coils
                finufft_exec!(plan1, r_t .* dcf_d, g_r_tmp)
                g_r .= sum(reshape(g_r_tmp, size(c_d)) .* conj(c_d), dims=4); 
            else
                finufft_exec!(plan1, r_t .* dcf_d, g_r)
            end
            # for line search, we can use that (NU)FFT is a linear operator, so do NUFFT only once:
            #w_d = s0 * e ^ (-t / T2*)
            g_r_reshaped = reshape(g_r, size(t2_d))

            g_s0 = g_r_reshaped .* exp.(-t_ms ./ t2_d)
            g_t2 = g_r_reshaped .* s0_d .* exp.(-t_ms ./ t2_d) .* (t_ms ./ t2_d.^2)

            finufft_exec!(plan2, reshape(g_s0, size(t2_d)) .* c_d, u_s0)
            finufft_exec!(plan2, reshape(g_t2, size(t2_d)) .* c_d, u_t2)

            u_s0 .*= dcf_d;
            u_t2 .*= dcf_d;

            for _ = 1:max_ls 
                alpha_t2 *= beta
                tmp_t2 .= r_t .- alpha_t2 * u_t2;
                obj = real(tmp_t2[:]' * tmp_t2[:]) / 2.0
                obj < obj0 && break
            end

            for _ = 1:max_ls 
                alpha_s0 *= beta
                tmp_s0 .= r_t .- alpha_s0 * u_s0;
                obj = real(tmp_s0[:]' * tmp_s0[:]) / 2.0
                obj < obj0 && break
            end

            # do the actual update
            t2_d .-= alpha_t2 * reshape(g_t2, size(t2_d));
            s0_d .-= alpha_s0 * reshape(g_s0, size(s0_d));

            finufft_exec!(plan2, w_d_t .* c_d, r_t)
            r_t .*= dcf_d;
            r_t .-= y_d_t;

            @info "it = $it, alpha_s0 = $alpha_s0, alpha_t2 = $alpha_t2, obj = $obj"
        end

        finufft_destroy!(plan1)
        finufft_destroy!(plan2)

    end

    # collect results from GPU & return: 
    t2_d

end
