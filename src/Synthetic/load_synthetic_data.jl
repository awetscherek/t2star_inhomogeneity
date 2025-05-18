function load_synthetic_data(eval_no, forward_operator, nchan)
    #function modified later to add noise to kspace
    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0.cfl")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0.cfl"))
        @info "Generating Synthetic Data"

        call_dynamic(eval_no, nchan)
    end

    r2 = 1 ./ Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2"))
    s0 = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0"))
    b0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0"))

    im = -γ .* b0
    e = complex.(r2, im)

    return forward_operator(e, s0)
end

function call_dynamic(x,a)
    name = Symbol("gen_phantom_$x")
    f = getfield(@__MODULE__, name)
    return f(x,a)
end