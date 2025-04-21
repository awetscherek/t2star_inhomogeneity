using FINUFFT

function forward_operator_intermediate_image_test(config, kx, ky, raw, time_since_last_rf, dims; # keyword arguments: 
    combine_coils = false,      # whether to use coil sensitivities
    sens = nothing,             # coil sensitivities ...
    use_dcf = false,            # whether to use pre-conditioner
    tol = 1e-9,                 # tolerance for FINUFFT
    timepoint_window_size=536)

    @assert !combine_coils || !isnothing(sens) "if we want to combine coils we need coil sensitivities ..."

    #kx and ky should be of shape
    #(269,8,536)
    # nky => necho => nkx
    nx, ny = dims;
    nz = size(raw, 4) # this assumes only data from one echo is passed to the function, but several slices, so raw should be a 4D array

    nkx, _, nky,_ = size(kx)

    @assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

    # this preconditioner could help speed up convergence:
    dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
    dcf = dcf ./ maximum(dcf)

    c_d = combine_coils ? sens : [1.0]; # this shouldn't make a copy of sens

    # use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
    kx_d = reshape(permutedims(kx, [2 1 3 4]) * config["FOVx"] / nx * 2 * pi, :, nky);
    ky_d = reshape(permutedims(ky, [2 1 3 4]) * config["FOVy"] / ny * 2 * pi, :, nky);

    # and use only data from central k-space region:
    selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi;

    # Considering the phase equation:
    # S(t) = S(0) .* exp(i .* gamma .* B_0 .* t - (t / T2*) )
    # Consider the exponent as e = (t / T2*) - i .* gamma .* B_0 .* t
    # Real{e} = (1 / T2*)
    # Im{e} = - gamma .* \delta B_0
    # such that exp(- t * e) = exp(i .* gamma .* B_0 .* t - (t / T2*) )

    e_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
    s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

    # this is the raw data from which we want to reconstruct the coil images
    #(timepoints, ky, nz * nchan)

    y_d = reshape(ComplexF64.(permutedims(raw,[3 1 5 4 2])) .* sqrt.(dcf), config["necho"] * nkx, :, nz * config["nchan"])[selection, :];

    dcf_d = use_dcf ? reshape(repeat(sqrt.(dcf), outer = (1, size(ky, 2))), :)[selection] : 1.0;

    total_timepoints = config["necho"] * nkx
    timepoints = ceil(Int, total_timepoints / timepoint_window_size)

    #Benchmark of forward operator using  E and S0 mappings from Intermediate Generation
    comb = combine_coils ? "" : "_no_combine_coils"
    e_d .= ComplexF64.(1 ./ (ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/Results/Intermediate/2d/t2$comb")))
    s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/Results/Intermediate/2d/s0$comb"))

    time_since_last_rf = vec(time_since_last_rf)

    # plan NUFFTs:
    plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

    r = Array{ComplexF64}(undef,size(y_d));
    r .= forward_operator(plan2, e_d, s0_d, timepoints, total_timepoints, kx_d, ky_d, c_d, time_since_last_rf, selection, timepoint_window_size)
    r .*= dcf_d;
    r .-= y_d;

    obj = real(r[:]' * r[:]) / 2.0; # objective function

    initial_obj = "Obj = $obj"
    @info initial_obj
end

function forward_operator(plan2, e_d, s0_d, timepoints, total_timepoints, kx_d, ky_d,
    c_d, time_since_last_rf, selection, timepoint_window_size)
    y_list = Vector{Array{ComplexF64}}(undef, timepoints)
    for t in 1:timepoints
        @info "t=$t"
        t_start = (t-1) * timepoint_window_size + 1
        t_end = min(t * timepoint_window_size, total_timepoints)
        approx_t = round(Int, (t_start + t_end) / 2)

        t_ms = time_since_last_rf[approx_t]

        sel = selection[t_start:t_end, :]
        kx_d_t = collect(kx_d[t_start:t_end,:][sel])
        ky_d_t = collect(ky_d[t_start:t_end,:][sel])

        finufft_setpts!(plan2, kx_d_t, ky_d_t)

        w_d_t = s0_d .* exp.(- t_ms .* e_d)

        y_t = finufft_exec(plan2, w_d_t .* c_d)
        
        y_list[t] = y_t
    end
    y = vcat(y_list...)

    return y
end