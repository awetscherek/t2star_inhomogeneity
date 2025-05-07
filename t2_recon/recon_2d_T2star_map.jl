includet("../operators/forward_operator.jl")
includet("../operators/adjoint_operator.jl")
includet("../approximate_time.jl")

using FINUFFT
using Optimisers
using ProgressBars
using Optim

const γ = 2 * π * 42.576e6

function recon_2d_t2star_map(config, kx, ky, raw, timepoints, dims; # keyword arguments: 
    combine_coils=false,      # whether to use coil sensitivities
    sens=nothing,             # coil sensitivities ...
    use_dcf=false,            # whether to use pre-conditioner
    tol=1e-9,                 # tolerance for FINUFFT
    niter=use_dcf ? 10 : 100, # number of gradient descent iterations
    timepoint_window_size=536,  # number of samples within each timepoint approximation window
    fat_modulation=nothing) # consideration of fat and water

    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    #kx and ky should be of shape
    #(269,8,536)
    # nky => necho => nkx
    nx, ny = dims;
    nz = size(raw, 4) # this assumes only data from one echo is passed to the function, but several slices, so raw should be a 4D array

    nkx, _, nky, _ = size(kx)

    @assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

    c_d = combine_coils ? sens : [1.0] # this shouldn't make a copy of sens

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [2 1 3 4]) * config["FOVx"] / nx * 2 * pi, :, nky)
    ky_d = reshape(permutedims(ky, [2 1 3 4]) * config["FOVy"] / ny * 2 * pi, :, nky)

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi

    dcf_y = use_dcf ? reshape(sqrt.(dcf), 1, size(dcf, 1), 1, 1, 1) : dcf
    dcf_d = use_dcf ? repeat(sqrt.(dcf), outer=(size(ky, 2), size(ky, 3)))[selection] : 1.0

    y_d = reshape(ComplexF64.(permutedims(raw, [3 1 5 4 2])) .* dcf_y, config["necho"] * nkx, :, nz * config["nchan"])[selection, :]

    num_total_timepoints = config["necho"] * nkx
    num_timepoints = ceil(Int, num_total_timepoints / timepoint_window_size)

    # Reshape fat modulation so it can be easily multiplied in k-space
    if !isnothing(fat_modulation)
        fat_modulation = repeat(vec(fat_modulation), 1, nky)[selection]
        fat_modulation .+= 1
    end

    # Considering the phase equation:
    # S(t) = S(0) .* exp(i .* γ .* Δb0 .* t - (t / T2*) )
    # Consider the exponent (t / T2*) - i .* γ .* Δb0 .* t
    # Such that S(t) = S(0) .* exp(e_d)
    # We use a variable e such that
    # Real{e} = (1 / T2*)
    # Im{e} = - γ .* Δb0
    # e = (1/T2* - i .* γ .* Δb0)
    # Then, exp(- t * e) = exp(i .* γ .* Δb0 .* t - (t / T2*) )

    e_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
    s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])

    # Take initial value of S0 to be reconstruction
    # s0_d .= 0.0;
    s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x_dcf")[:, :, :, 1])
    # s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/s0_dcf"));

    r2 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])
    Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    #im = im{e} = - γ .* Δb0
    im = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    r2 .= 1 / 50.0
    Δb0 .= 0
    # Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$ip_dcf"))

    im = -γ .* Δb0

    e_d .= complex.(r2, im)

    #intermediate result, required for gradient at each time point
    # g_e = combine_coils ? Array{ComplexF64}(undef, size(e_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"])
    # g_s0 = combine_coils ? Array{ComplexF64}(undef, size(s0_d)) : Array{ComplexF64}(undef, nx, ny, nz * config["nchan"])

    # plan NUFFTs:
    plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    r = Array{ComplexF64}(undef, size(y_d))

    function flatten(X,Y)
        return vcat(vec(X), vec(Y))
    end

    function unflatten(X)
        N = length(X) ÷ 2
        return reshape(X[1:N], size(e_d)), reshape(X[N+1:end], size(s0_d))
    end

    # Initialise Operators with implicit values
    function forward_operator(x)
        e, s0 = unflatten(x)
        r .= forward_operator_impl(plan2, e, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
        timepoint_window_size, fat_modulation)
        r .*= dcf_d
        r .-= y_d
        obj = 1/2 * sum(abs2, r)
        @info "obj = $obj"
        return obj
    end

    function adjoint_operator!(storage, x)
        e, s0 = unflatten(x)
        g_e, g_s0 = adjoint_operator_impl(plan1, r, e, s0, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
        timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, nx, ny, nz, config["nchan"])
        storage .= flatten(g_e, g_s0)
    end

    # r .= forward_operator(flatten(e_d, s0_d))

    # r .*= dcf_d
    # r .-= y_d
    # obj = 1/2* sum(abs2, r)
    # obj = real(r[:]' * r[:]) / 2.0 # objective function
    obj = forward_operator(flatten(e_d, s0_d))

    initial_obj = "Initial obj = $obj"
    @info initial_obj

    open("output.txt", "a") do f
        println(f, string(initial_obj))
    end

    # Optimiser
    # model = (S0=s0_d, e=e_d)
    # state = Optimisers.setup(Optimisers.AdamW(), model)

    initial_guess = flatten(e_d, s0_d)

    # iter = ProgressBar(1:niter)
    # for it in iter
    #     g_e, g_s0 = adjoint_operator(e_d, s0_d)

    #     gradients = (S0=g_s0, e=g_e)
    #     state, model = Optimisers.update(state, model, gradients)
    #     s0_d, e_d = model.S0, model.e

    #     r .= forward_operator(e_d, s0_d)

    #     r .*= dcf_d
    #     r .-= y_d

    #     obj = real(r[:]' * r[:]) / 2.0

    #     info = "it = $it, obj = $obj"
    #     # @info info
    #     open("output.txt", "a") do f
    #         println(f, string(info))
    #     end

    #     set_description(iter, "obj: $obj")
    # end

    results = optimize(forward_operator, adjoint_operator!,
        initial_guess,
        LBFGS(),
        Optim.Options(
            iterations = niter,
            show_trace = true))

    x = Optim.minimizer(results)

    println("x")
    println(size(x))

    e_d, s0_d = unflatten(x)

    finufft_destroy!(plan1)
    finufft_destroy!(plan2)

    # Im{e} = - γ .* Δb0
    # Δb0 = - Im{e} ./ γ
    Δb0 = imag(e_d) ./ (-γ)

    # collect results from GPU & return: 
    1 ./ real(e_d), s0_d, Δb0
end