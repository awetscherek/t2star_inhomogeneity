function load_synthetic_data(eval_no, forward_operator)
    #function modified later to add noise to kspace
    if (!isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2") 
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0")
        || !isfile("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0"))
        @info "Generating Synthetic Data"

        call_dynamic(eval_no)
    end

    r2 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2"))
    s0 = ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0"))
    b0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0"))

    im = -γ .* b0
    e = complex.(r2, im)

    return forward_operator(e, s0)
end

function call_dynamic(x)
    name = Symbol("gen_synthetic_data_$x")
    f = getfield(@__MODULE__, name)
    return f()
end