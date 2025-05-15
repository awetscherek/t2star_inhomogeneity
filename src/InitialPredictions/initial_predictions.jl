abstract type Mode end
struct Synthetic <: Mode end
struct Real      <: Mode end


function initialise_params(::Synthetic, eval_no, e_d, s0_d)
    initialise_synthetic_params(eval_no, e_d, s0_d)
end

function initialise_params(::Synthetic, eval_no, e_d, s0_fat, s0_water)
    initialise_synthetic_params(eval_no, e_d, s0_fat, s0_water)
end

function initialise_params(::Real, e_d, s0_d)
    initialise_real_params(e_d, s0_d)
end

function initialise_params(::Real, e_d, s0_fat, s0_water)
    initialise_real_params(e_d, s0_fat, s0_water)
end