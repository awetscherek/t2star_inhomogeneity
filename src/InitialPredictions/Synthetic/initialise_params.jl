function initialise_synthetic_params(eval_no, e_d, s0_d)
    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    r2_d .= 1.0 / 50

    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_$eval_no.cfl"))

    im = -γ .* b0_d
    e_d .= complex.(r2_d, im)

    s0_d .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no"))
end

function initialise_synthetic_params(eval_no, e_d, s0_fat, s0_water)
    r2_d =  Array{Float64}(undef, size(e_d))
    b0_d =  Array{Float64}(undef, size(e_d))

    r2_d .= 1.0 / 50

    b0_d .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/b0_$eval_no.cfl"))

    im = -γ .* b0_d
    e_d .= complex.(r2_d, im)

    s0_fat .= ComplexF64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/s0_$eval_no")) ./ 2;
    s0_water .= s0_fat
end