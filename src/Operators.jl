# struct params
#     s0

# end

# struct gradients
# end

include("approximate_time.jl")

# Phase equation:
# S(t) = S(0) .* exp(i .* γ .* Δb0 .* t - (t / T2*) )

include("Operators/forward_operator.jl")
include("Operators/adjoint_operator.jl")