function image_recon_3d(config, kx, ky, kz, raw, dims; # keyword arguments: 
    combine_coils = false,      # whether to use coil sensitivities
    sens = nothing,             # coil sensitivities ...
    use_dcf = false,            # whether to use pre-conditioner
    tol = 1e-9,                 # tolerance for FINUFFT
    niter = use_dcf ? 10 : 100, # number of gradient descent iterations
    max_ls = 100,               # number of line search iterations
    alpha0 = 1.0,               # initial step size for line search
    beta = 0.6)                 # factor to decrease step size

    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    nx, ny, nz = dims;

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

    c_d = combine_coils ? CuArray(sens) : CuArray([1.0]);

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(CuArray(kx * config["FOVx"] / nx * 2 * pi), :);
    ky_d = reshape(CuArray(ky * config["FOVy"] / ny * 2 * pi), :);
    kz_d = reshape(CuArray(kz * config["FOVz"] / nz * 2 * pi), :);

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi .&& -pi .<= kz_d .< pi;

    # this is where the coil images will be stored:
    x_d = combine_coils ? CuArray{ComplexF64}(undef, nx, ny, nz) : CuArray{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

    # this is the raw data from which we want to reconstruct the coil images (first echo, coils last):
    y_d = reshape(CuArray(ComplexF64.(permutedims(raw, [1 3 4 2])) .* sqrt.(dcf)), :, config["nchan"])[selection, :];

    dcf_d = use_dcf ? reshape(CuArray(repeat(sqrt.(dcf), outer = (1, size(ky, 2), size(ky, 3), size(ky, 4)))), :)[selection] : 1.0;

    # plan NUFFTs:
    plan2 = cufinufft_makeplan(2, dims, 1, config["nchan"], tol)     # type 2 (forward transform)
    cufinufft_setpts!(plan2, kx_d[selection], ky_d[selection], kz_d[selection])
    plan1 = cufinufft_makeplan(1, dims, -1, config["nchan"], tol)    # type 1 (adjoint transform)
    cufinufft_setpts!(plan1, kx_d[selection], ky_d[selection], kz_d[selection])

    # allocate some arrays:
    r = CuArray{ComplexF64}(undef, size(y_d));
    g = CuArray{ComplexF64}(undef, size(x_d));
    u = CuArray{ComplexF64}(undef, size(y_d));
    tmp = CuArray{ComplexF64}(undef, size(y_d));

    gtmp = CuArray{ComplexF64}(undef, nx, ny, nz, config["nchan"]); # intermediate results, needed to combine gradient across coils...

    # initial guess:
    x_d .= 0.0;

    # calculate the residual
    cufinufft_exec!(plan2, x_d .* c_d, r)
    r .*= dcf_d;
    r .-= y_d;
    obj = real(r[:]' * r[:]) / 2.0; # objective function
    alpha = alpha0 * beta
    @info "obj0 = $obj"
    for it = 1:niter
        alpha /= beta # otherwise step size alpha decreases too fast ... 
        obj0 = obj
        if combine_coils
            cufinufft_exec!(plan1, r .* dcf_d, gtmp)  # gradient
            g .= sum(gtmp .* conj(c_d), dims=4);   
        else
            cufinufft_exec!(plan1, r .* dcf_d, g)
        end
        cufinufft_exec!(plan2, g .* c_d, u)  # for line search, we can use that (NU)FFT is a linear operator, so do NUFFT only once:
        u .*= dcf_d;

        for _ = 1:max_ls 
            alpha *= beta
            tmp .= r .- alpha * u;
            obj = real(tmp[:]' * tmp[:]) / 2.0
            obj < obj0 && break
        end

        x_d .-= alpha * g; # do the actual update
        cufinufft_exec!(plan2, x_d .* c_d, r)
        r .*= dcf_d;
        r .-= y_d;

        @info "it = $it, alpha = $alpha, obj = $obj"
    end

    cufinufft_destroy!(plan1)
    cufinufft_destroy!(plan2)

    # collect results from GPU & return: 
    Array(x_d)

end
