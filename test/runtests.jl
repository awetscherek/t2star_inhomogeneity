using DqT2
using Test
using FINUFFT

include("forward_op_consistency.jl")
include("zero_filled_guess.jl")

#------------------------------------
# Initialise Test Data 

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation)

dims = [nx, ny]
tol=1e-9

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

(
    y_d,
    kx_d,
    ky_d,
    dcf_d,
    c_d,
    selection,
    num_timepoints,
    num_total_timepoints,
    fat_modulation
) = preprocess_data(
    config,
    raw,
    combine_coils,
    sens,
    kx,
    ky,
    timepoint_window_size,
    use_dcf,
    fat_modulation
    )

# plan NUFFTs:
plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

r = Array{ComplexF64}(undef, size(y_d))

#------------------------------------------------------
#Setup test functions

function flatten(X,Y)
    return vcat(vec(X), vec(Y))
end

function unflatten(X)
    N = length(X) ÷ 2
    return reshape(X[1:N], size(e_d)), reshape(X[N+1:end], size(s0_d))
end

# Initialise Operators with implicit values
function forward_operator(e,fat,water)
    r .= forward_operator_impl(plan2, e, fat,water, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
    timepoint_window_size, fat_modulation)

    r .*= dcf_d
    r .-= y_d
    obj = 1/2 * sum(abs2, r)
    @info "obj = $obj"
    return obj
end

function forward_operator(e,s0)
    r .= forward_operator_impl(plan2, e, nothing, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
    timepoint_window_size, fat_modulation)
    
    r .*= dcf_d
    r .-= y_d
    obj = 1/2 * sum(abs2, r)
    @info "obj = $obj"
    return obj
end

# function adjoint_operator!(r2, b0, fat, water)
#     return adjoint_operator_impl(plan1, r, r2, b0, fat, water, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
#     timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation,config["nchan"])
# end

function adjoint_operator(e, s0)
    return adjoint_operator_impl(plan1, r, e, s0, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
    timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, config["nchan"])
end

# zero_filled_guess(combine_coils, config, forward_operator)

# @testset "Forward/Adjoint gradient consistency" begin
#     res = (nx, ny, nz)
#     check_forward_adjoint_gradient_consistency(
#       forward_operator,
#       adjoint_operator,
#       res,   # r2
#       res,   # b0
#       res;   # s0
#       rtol=0.05,
#       ε=5e-2,
#       repetitions=30,
#     )
# end

@testset "Forward/Adjoint gradient consistency" begin
    res = (nx, ny, nz)
    check_forward_adjoint_gradient_consistency_e(
      forward_operator,
      adjoint_operator,
      res,   # r2
      res;   # s0
      rtol=0.05,
      ε=5e-2,
      repetitions=30,
    )
end