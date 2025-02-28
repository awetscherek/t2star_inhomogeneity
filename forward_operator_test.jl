using FINUFFT
using ReadWriteCFL

function forward_operator_test(config, kx, ky, raw, time_since_last_rf, dims; # keyword arguments: 
    combine_coils = false,      # whether to use coil sensitivities
    sens = nothing,             # coil sensitivities ...
    use_dcf = false,            # whether to use pre-conditioner
    tol = 1e-9,                 # tolerance for FINUFFT
    niter = use_dcf ? 10 : 100, # number of gradient descent iterations
    max_ls = 100,               # number of line search iterations
    alpha0 = 1.0,               # initial step size for line search
    beta = 0.05)                 # factor to decrease step size

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

    c_d = combine_coils ? sens : [1.0];

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

    if combine_coils
        t2_d = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/t2star_2d")
        s0_d = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/s0_2d")
    else
        t2_d = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/t2star_2d_no_combine_coils")
        s0_d = ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/s0_2d_no_combine_coils")
    end
    t2_d = ComplexF64.(t2_d)
    s0_d = ComplexF64.(s0_d)

    y_d = reshape(ComplexF64.(permutedims(raw,[1 3 5 4 2])) .* sqrt.(dcf), nkx * config["necho"], :, nz * config["nchan"])[selection, :];

    dcf_d = use_dcf ? reshape(repeat(sqrt.(dcf), outer = (1, size(ky, 2))), :)[selection] : 1.0;

    timepoints = config["necho"] * nkx

    time_since_last_rf = vec(time_since_last_rf)

    println("For the plan1 nufft")
    println("kx_d")
    println(size(kx_d[selection]))
    println("ky_d")
    println(size(ky_d[selection]))

    println("size of r should be the same as y_d which is")
    println(size(y_d))
    r = forward_operator(t2_d, s0_d, timepoints, kx_d, ky_d, nz * config["nchan"], c_d, tol, time_since_last_rf, selection, dims)
    r .*= dcf_d;
    r .-= y_d;
    obj = real(r[:]' * r[:]) / 2.0; # objective function
    @info "obj0 = $obj"
end

function forward_operator(t2_d, s0_d, timepoints, kx_d, ky_d, ntrans, c_d, tol, time_since_last_rf, selection, dims)
    r_list = Vector{Array{ComplexF64}}(undef, timepoints)
    for t in 1:timepoints
        @info "t=$t"
        t_ms = time_since_last_rf[t]

        kx_d_t = collect(kx_d[t,:][selection[t,:]])
        ky_d_t = collect(ky_d[t,:][selection[t,:]])

        # println("kx_d_t")
        # println(size(kx_d_t))
        # println("ky_d_t")
        # println(size(ky_d_t))

        plan2 = finufft_makeplan(2, dims, 1, ntrans, tol)     # type 2 (forward transform)
        finufft_setpts!(plan2, kx_d_t, ky_d_t)

        # calculate the residual
        w_d_t = s0_d .* exp.(t_ms ./ t2_d)

        r_t = finufft_exec(plan2, w_d_t .* c_d)
        # r_t = finufft_exec(plan2, w_d_t)
        
        r_list[t] = r_t

        finufft_destroy!(plan2)
    end
    r = vcat(r_list...)
    println("size of r")
    println(size(r))
    return r
end
