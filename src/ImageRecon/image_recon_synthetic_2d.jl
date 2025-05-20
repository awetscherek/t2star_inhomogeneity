function image_recon_synthetic_2d(config, kx, ky, y_d; # keyword arguments: 
    combine_coils = false,      # whether to use coil sensitivities
    sens = nothing,             # coil sensitivities ...
    use_dcf = false,            # whether to use pre-conditioner
    tol = 1e-9,                 # tolerance for FINUFFT
    niter = use_dcf ? 10 : 100, # number of gradient descent iterations
    max_ls = 100,               # number of line search iterations
    alpha0 = 1.0,               # initial step size for line search
    beta = 0.6)                 # factor to decrease step size

    dims = [nx,ny]

    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

    c_d = combine_coils ? sens : [1.0]; # this shouldn't make a copy of sens

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [2 1 3]) * config["FOVx"] / nx * 2 * pi, :);
    ky_d = reshape(permutedims(ky, [2 1 3]) * config["FOVy"] / ny * 2 * pi, :);

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi;

    # this is where the coil images will be stored - note that we still will reconstruct several slices:
    x_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

    dcf_d = use_dcf ? reshape(repeat(sqrt.(dcf), outer = (1, size(ky, 2))), :)[selection] : 1.0;

    # plan NUFFTs:
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)
    finufft_setpts!(plan2, kx_d[selection], ky_d[selection])
    plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
    finufft_setpts!(plan1, kx_d[selection], ky_d[selection])

    # allocate some arrays:
    r = Array{ComplexF64}(undef, size(y_d));
    g = combine_coils ? Array{ComplexF64}(undef, size(x_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]);
    u = Array{ComplexF64}(undef, size(y_d));
    tmp = Array{ComplexF64}(undef, size(y_d));

    gtmp = Array{ComplexF64}(undef, nx, ny, nz * config["nchan"]); # intermediate results, needed to combine gradient across coils...

    # initial guess:
    x_d .= 0.0;

    # calculate the residual
    finufft_exec!(plan2, x_d .* c_d, r)
    r .*= dcf_d;
    r .-= y_d;
    obj = real(r[:]' * r[:]) / 2.0; # objective function
    alpha = alpha0 * beta
    @info "obj0 = $obj"
    for it = 1:niter
        alpha /= beta # otherwise step size alpha decreases too fast ... 
        obj0 = obj
        if combine_coils
            finufft_exec!(plan1, r .* dcf_d, gtmp)  # gradient
            g .= sum(reshape(gtmp, size(c_d)) .* conj(c_d), dims=4);   
        else
            finufft_exec!(plan1, r .* dcf_d, g)
        end
        finufft_exec!(plan2, reshape(g, size(x_d)) .* c_d, u)  # for line search, we can use that (NU)FFT is a linear operator, so do NUFFT only once:
        u .*= dcf_d;

        for _ = 1:max_ls 
            alpha *= beta
            tmp .= r .- alpha * u;
            obj = real(tmp[:]' * tmp[:]) / 2.0
            obj < obj0 && break
        end

        x_d .-= alpha * reshape(g, size(x_d)); # do the actual update
        finufft_exec!(plan2, x_d .* c_d, r)
        r .*= dcf_d;
        r .-= y_d;

        @info "it = $it, alpha = $alpha, obj = $obj"
    end

    finufft_destroy!(plan1)
    finufft_destroy!(plan2)

    # collect results from GPU & return: 
    x_d

end
