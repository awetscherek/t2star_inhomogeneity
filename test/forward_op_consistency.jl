using Random, Test
using Random: Xoshiro
using LinearAlgebra

# """
#     check_forward_adjoint_gradient_consistency(
#       forward_op, adjoint_op,
#       dims_r2::NTuple{N,Int}, dims_b0::NTuple{M,Int}, dims_s0::NTuple{K,Int};
#       rtol=0.05, ε=5e-2, repetitions=30,
#     )

# Verifies that `adjoint_op(r2,b0,s0)` really computes the gradient of the scalar-output
# `forward_op(r2,b0,s0)` by comparing directional derivatives against finite differences.

# - `dims_r2`, `dims_b0` are the shapes of your *real* parameter arrays.
# - `dims_s0` is the shape of your *complex* parameter array.
# - `rtol` is the relative‐tolerance for `isapprox`.
# - `ε` is the FD step‐size.
# - `repetitions` is how many random one‐hot directions to test.
# """
# function check_forward_adjoint_gradient_consistency(
#     forward_op::Function,
#     adjoint_op::Function,
#     dims_r2::NTuple{N,Int} where N,
#     dims_b0::NTuple{M,Int} where M,
#     dims_s0::NTuple{K,Int} where K;
#     rtol::Float64=0.05,
#     ε::Float64=5e-2,
#     repetitions::Int=30,
# )
#     rng = Xoshiro(42)

#     # 1) pick a random test point
#     r2 = rand(rng, Float64, dims_r2...)
#     b0 = rand(rng, Float64, dims_b0...)
#     s0 = rand(rng, ComplexF64, dims_s0...)

#     # 2) forward: scalar objective
#     f0 = forward_op(r2, b0, s0)
#     @test isa(f0, Real)

#     # 3) call adjoint to get gradients
#     grad_r2, grad_b0, grad_s0 = adjoint_op(r2, b0, s0)

#     # 4) compare directional derivatives
#     N1, N2, N3 = length(r2), length(b0), length(s0)
#     total = N1 + N2 + N3

#     graddirs   = Float64[]
#     dirapproxs = Float64[]

#     for _ in 1:repetitions
#         # create one‐hot direction in the concatenated space
#         dir_r2 = zeros(Float64, dims_r2...)
#         dir_b0 = zeros(Float64, dims_b0...)
#         dir_s0 = zeros(ComplexF64, dims_s0...)
#         idx = rand(rng, 1:total)
#         if idx ≤ N1
#             dir_r2[idx] = one(Float64)
#         elseif idx ≤ N1 + N2
#             dir_b0[idx - N1] = one(Float64)
#         else
#             dir_s0[idx - N1 - N2] = one(ComplexF64)
#         end

#         # exact directional derivative via inner‐product
#         dd_r2 = sum(dir_r2 .* grad_r2)
#         dd_b0 = sum(dir_b0 .* grad_b0)
#         dd_s0 = real(sum(conj(dir_s0) .* grad_s0))  # complex part
#         push!(graddirs, dd_r2 + dd_b0 + dd_s0)

#         # finite-difference approximation
#         fp = forward_op(r2 .+ ε .* dir_r2,
#                         b0 .+ ε .* dir_b0,
#                         s0 .+ ε .* dir_s0)
#         fm = forward_op(r2 .- ε .* dir_r2,
#                         b0 .- ε .* dir_b0,
#                         s0 .- ε .* dir_s0)
#         push!(dirapproxs, (fp - fm) / (2ε))
#     end

#     @info "relative error = " *
#             string(norm(graddirs - dirapproxs) /
#                     max(norm(graddirs), norm(dirapproxs)))
                    
#     @test isapprox(graddirs, dirapproxs; rtol=rtol)
# end


"""
    check_forward_adjoint_gradient_consistency(
      forward_op, adjoint_op,
      dims_r2::NTuple{N,Int}, dims_b0::NTuple{M,Int}, dims_s0::NTuple{K,Int};
      rtol=0.05, ε=5e-2, repetitions=30,
    )

Verifies that `adjoint_op(r2,b0,s0)` really computes the gradient of the scalar-output
`forward_op(r2,b0,s0)` by comparing directional derivatives against finite differences.

- `dims_r2`, `dims_b0` are the shapes of your *real* parameter arrays.
- `dims_s0` is the shape of your *complex* parameter array.
- `rtol` is the relative‐tolerance for `isapprox`.
- `ε` is the FD step‐size.
- `repetitions` is how many random one‐hot directions to test.
"""
function check_forward_adjoint_gradient_consistency_e(
    forward_op::Function,
    adjoint_op::Function,
    dims_e::NTuple{N,Int} where N,
    dims_s0::NTuple{K,Int} where K;
    rtol::Float64=0.05,
    ε::Float64=5e-2,
    repetitions::Int=30,
)
    rng = Xoshiro(42)

    # 1) pick a random test point
    e = rand(rng, ComplexF64, dims_e...)
    s0 = rand(rng, ComplexF64, dims_s0...)

    # 2) forward: scalar objective
    f0 = forward_op(e, s0)
    @test isa(f0, Real)

    # 3) call adjoint to get gradients
    grad_e, grad_s0 = adjoint_op(e, s0)

    # 4) compare directional derivatives
    N1, N2 = length(e), length(s0)
    total = N1 + N2

    graddirs   = Float64[]
    dirapproxs = Float64[]

    for _ in 1:repetitions
        # create one‐hot direction in the concatenated space
        dir_e = zeros(ComplexF64, dims_e...)
        dir_s0 = zeros(ComplexF64, dims_s0...)
        idx = rand(rng, 1:total)
        if idx ≤ N1
            dir_e[idx] = one(ComplexF64)
        else
            dir_s0[idx - N1] = one(ComplexF64)
        end

        # exact directional derivative via inner‐product
        dd_e = real(sum(conj(dir_e) .* grad_e)) 
        dd_s0 = real(sum(conj(dir_s0) .* grad_s0))  # complex part
        push!(graddirs, dd_e + dd_s0)

        # finite-difference approximation
        fp = forward_op(e .+ ε .* dir_e,
                        s0 .+ ε .* dir_s0)
        fm = forward_op(e .- ε .* dir_e,
                        s0 .- ε .* dir_s0)
        push!(dirapproxs, (fp - fm) / (2ε))
    end

    @info "relative error = " *
            string(norm(graddirs - dirapproxs) /
                    max(norm(graddirs), norm(dirapproxs)))
                    
    @test isapprox(graddirs, dirapproxs; rtol=rtol)
end