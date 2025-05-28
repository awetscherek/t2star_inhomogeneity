module SyntheticPhantoms

using ImageFiltering

const γ = 2 * π * 42.576e6

export circle_phantom,
       concentric_circles_phantom,
       checkerboard_phantom,
       radial_rings_phantom,
       linear_gradient_phantom,
       smooth_random_phantom,
       shepp_logan_phantom

function apply_phase!(S0::Array{ComplexF64}, B0, xs, ys; φ0=0.0, φx=0.0, φy=0.0, TE0=0.0)
    # φ0: global phase offset (rad)
    # φx, φy: spatial ramp (rad per unit length)
    # TE0: first echo time (s)
    nx, ny, nz = size(S0)
    for k in 1:nz, j in 1:ny, i in 1:nx
        phase_ramp = φx*xs[i] + φy*ys[j]
        total_phase = φ0 + γ*B0[i,j,k]*TE0 + phase_ramp
        S0[i,j,k] *= exp(1im*total_phase)
    end
end

function apply_phase_fatmod!(fat::Array{ComplexF64}, water::Array{ComplexF64}, B0, xs, ys; φ0=0.0, φx=0.0, φy=0.0, TE0=0.0)
    # φ0: global phase offset (rad)
    # φx, φy: spatial ramp (rad per unit length)
    # TE0: first echo time (s)
    nx, ny, nz = size(fat)
    for k in 1:nz, j in 1:ny, i in 1:nx
        phase_ramp = φx*xs[i] + φy*ys[j]
        total_phase = φ0 + γ*B0[i,j,k]*TE0 + phase_ramp
        fat[i,j,k] *= exp(1im*total_phase)
        water[i,j,k] *= exp(1im*total_phase)
    end
end

"""
Generate a circular phantom with complex S0 phase:
"""
function circle_phantom(nx, ny, nz; R=0.6, TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)
    S0  = zeros(ComplexF64, nx, ny, nz)
    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        r = sqrt(xs[i]^2 + ys[j]^2)
        if r <= R
            S0[i,j,k] = 1.0 + 0im
            T2s[i,j,k] = 50.0
        end
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Checkerboard pattern with complex S0 phase:
"""
function checkerboard_phantom(nx, ny, nz; nblocks=8, b0_min::Float64=-550.0/(1000*γ), b0_max::Float64= 550.0/(1000*γ),
                                                    TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(0.0, 1.0, nx)
    ys = LinRange(0.0, 1.0, ny)
    S0  = fill(1.0 + 0.0im, nx, ny, nz)
    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        bi = floor(Int, xs[i]*nblocks)
        bj = floor(Int, ys[j]*nblocks)
        T2s[i,j,k] = (isodd(bi + bj) ? 30.0 : 80.0)

        # Left-right gradient for B0
        B0[i,j,k]  = b0_min + (b0_max - b0_min) * (j - 1) / (ny - 1)
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Linear gradient phantom with complex S0 phase:
"""
function linear_gradient_phantom(nx, ny, nz; Gx=500.0/(1000 * γ), Gy=-300.0/(1000 * γ), TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)
    S0  = fill(1.0 + 0.0im, nx, ny, nz)
    T2s = fill(5.0, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        B0[i,j,k] = Gx*xs[i] + Gy*ys[j]
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Linear gradient phantom of T2 left-right and B0 top-bottom:
"""
function linear_gradient_t2_b0(nx, ny, nz;
                               t2_min::Float64=0.0, t2_max::Float64=100.0,
                               b0_min::Float64=-550.0/(1000*γ), b0_max::Float64= 550.0/(1000*γ),
                               TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)

    # pre‐allocate
    S0  = fill(1.0 + 0.0im, nx, ny, nz)
    T2s = zeros(Float64,  nx, ny, nz)
    B0  = zeros(Float64,  nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        # Top-bottom gradient for T2
        T2s[i,j,k] = t2_min + (t2_max - t2_min) * (i - 1) / (nx - 1)

        # Left-right gradient for B0
        B0[i,j,k]  = b0_min + (b0_max - b0_min) * (j - 1) / (ny - 1)
    end

    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Linear gradient phantom of T2 left-right and B0 top-bottom:
"""
function low_t2_high_b0(nx, ny, nz;
                               TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)

    # pre‐allocate
    S0  = fill(1.0 + 0.0im, nx, ny, nz)
    T2s = fill(2,  nx, ny, nz)
    B0  = fill(1000/(1000 * γ), nx, ny, nz)

    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

#-----------------------------------------------------
#Fatmod

"""
Generate a circular phantom with complex S0 phase:
"""
function circle_phantom_fatmod(nx, ny, nz;
                              R=0.6,
                              separation=0.4,
                              TE0=0.0,
                              φ0=0.0,
                              φx=0.0,
                              φy=0.0)

    # sample grid in x/y
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)

    # preallocate
    fat   = zeros(ComplexF64, nx, ny, nz)
    water = zeros(ComplexF64, nx, ny, nz)
    T2s   = zeros(Float64,    nx, ny, nz)
    B0    = zeros(Float64,    nx, ny, nz)

    # circle centers
    cx_fat,   cy_fat   = -separation/2, 0.0
    cx_water, cy_water =  separation/2, 0.0

    for k in 1:nz, j in 1:ny, i in 1:nx
        x, y = xs[i], ys[j]

        # distance to each center
        r_fat   = sqrt((x - cx_fat)^2   + (y - cy_fat)^2)
        r_water = sqrt((x - cx_water)^2 + (y - cy_water)^2)

        # set signals to 1 inside each disk
        if r_fat <= R
            fat[i,j,k] = 1.0 + 0im
        end
        if r_water <= R
            water[i,j,k] = 1.0 + 0im
        end

        # assign T2* in union of disks
        if r_fat <= R || r_water <= R
            T2s[i,j,k] = 50.0
        end
    end

    # apply your phase/fat-modulation step
    apply_phase_fatmod!(fat, water, B0, xs, ys;
                        φ0=φ0, φx=φx, φy=φy, TE0=TE0)

    return fat, water, T2s, B0
end

"""
Checkerboard pattern with complex S0 phase:
"""
function checkerboard_phantom_fatmod(nx, ny, nz; nblocks=8, TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(0.0, 1.0, nx)
    ys = LinRange(0.0, 1.0, ny)

    fat  = zeros(ComplexF64, nx, ny, nz)
    water  = zeros(ComplexF64, nx, ny, nz)

    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        bi = floor(Int, xs[i]*nblocks)
        bj = floor(Int, ys[j]*nblocks)
        T2s[i,j,k] = (isodd(bi + bj) ? 30.0 : 80.0)

        fat[i,j,k] = (isodd(bi + bj) ? 0 + 0im : 1 + 0im)
        water[i,j,k] = (isodd(bi + bj) ? 1 + 0im : 0 + 0im)

        B0[i,j,k]  = 20.0/(1000 * γ) * sin(2π*xs[i]) * sin(2π*ys[j])
    end
    apply_phase_fatmod!(fat,water, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return fat, water, T2s, B0
end

end # module
