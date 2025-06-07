function zero_filled_guess(combine_coils, config, forward_operator)
    s0_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
    e_d = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"])
    
    e_d .= 0 + 0im
    s0_d .= 0 + 0im

    forward_operator(e_d, s0_d)
end