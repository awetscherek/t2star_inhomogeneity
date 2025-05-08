"""
    check_norm_gradient_consistency(
        operator::Operator,
        forward_dimensions::Tuple{Vararg{Int}},
        rtol::Float32=0.03f0,
        epsilon::Float64=1e-1;
        repetitions::Int=20,
    )

Check that the gradient is consistent with its estimate using the Newton method from the objective function.
`rtol` controls the relative tolerance of the check.
`epsilon` controls the step size of the Newton method.
`repetitions` controls the number of random directions to check.
"""
function check_objective_gradient_consistency(
    operator::Operator{PT},
    forward_dimensions::Tuple{Vararg{Int}},
    rtol::Float32=0.05f0,
    epsilon::Float64=5e-2;
    repetitions::Int=30,
) where {PT}
    rng = Xoshiro(42)
    # d* gf(x) = 1 / 2e ((f(x + e*d) - f(x - e*d))) by relative error
    point = rand(rng, PT, forward_dimensions)
    grad = zero(point)
    (normvalue, _) = gradient!(operator, point, grad)
    # Check that the objective function and the objective function obtained when calculating the gradient are similar
    @test isapprox(normvalue, norm(operator, point))

    # Check that the gradient is consistent with Newton method
    graddirs = Vector{Float32}()
    dirapproxs = Vector{Float32}()
    for i in 1:repetitions
        direction = zeros(PT, forward_dimensions)
        direction[CartesianIndex(rand(rng, 1:prod(forward_dimensions)))] = one(PT) # Set a random direction to 1
        graddir = real(dot(direction, grad))
        push!(graddirs, graddir)
        dirapprox =
            (
                norm(operator, PT.(point .+ epsilon .* direction)) -
                norm(operator, PT.(point .- epsilon .* direction))
            ) / (2 * epsilon)
        push!(dirapproxs, dirapprox)
    end
    if !isapprox(graddirs, dirapproxs, rtol=rtol)
        println(operator, " ", norm(graddirs - dirapproxs) / max(norm(graddirs), norm(dirapproxs)))
    end
    @test isapprox(graddirs, dirapproxs, rtol=rtol)
end
