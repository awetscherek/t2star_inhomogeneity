function zero_filled_guess(combine_coils, config, forward_operator)
    r2_d = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])
    b0_d = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])
    s0_d = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"])

    r2_d .= 0.0
    b0_d .= 0.0
    s0_d .= 0.0

    forward_operator(r2_d, b0_d, s0_d)
end