module SyntheticPhantoms

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
    return S0
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
    S0 .= 1.0 + 0.0im
    for k in 1:nz, j in 1:ny, i in 1:nx
        r = sqrt(xs[i]^2 + ys[j]^2)
        if r <= R
            T2s[i,j,k] = 50.0
        end
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Two concentric tissue rings with complex S0 phase:
"""
function concentric_circles_phantom(nx, ny, nz; R1=0.3, R2=0.6, Gx=100.0, TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)
    S0  = zeros(ComplexF64, nx, ny, nz)
    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        r = sqrt(xs[i]^2 + ys[j]^2)
        if r <= R1
            S0[i,j,k]  = 1.0 + 0.0im
            T2s[i,j,k] = 40.0
        elseif r <= R2
            S0[i,j,k]  = 0.6 + 0.0im
            T2s[i,j,k] = 80.0
        end
        B0[i,j,k] = Gx * xs[i]
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Checkerboard pattern with complex S0 phase:
"""
function checkerboard_phantom(nx, ny, nz; nblocks=8, TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(0.0, 1.0, nx)
    ys = LinRange(0.0, 1.0, ny)
    S0  = fill(1.0 + 0.0im, nx, ny, nz)
    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        bi = floor(Int, xs[i]*nblocks)
        bj = floor(Int, ys[j]*nblocks)
        T2s[i,j,k] = (isodd(bi + bj) ? 30.0 : 80.0)
        B0[i,j,k]  = 20.0 * sin(2π*xs[i]) * sin(2π*ys[j])
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Radial rings with complex S0 phase:
"""
function radial_rings_phantom(nx, ny, nz; M=5, TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)
    S0  = zeros(ComplexF64, nx, ny, nz)
    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    ΔR = 1.0 / M
    for k in 1:nz, j in 1:ny, i in 1:nx
        r = sqrt(xs[i]^2 + ys[j]^2)
        ring = clamp(floor(Int, r/ΔR) + 1, 1, M)
        amp = 1.0 + 0.2*(ring - (M+1)/2)/((M-1)/2)
        S0[i,j,k]  = amp + 0.0im
        T2s[i,j,k] = 20.0 + 60.0*(ring-1)/(M-1)
        B0[i,j,k]  = 10.0 * cos(2π*r/ΔR)
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Linear gradient phantom with complex S0 phase:
"""
function linear_gradient_phantom(nx, ny, nz; Gx=50.0, Gy=-30.0, TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)
    S0  = fill(1.0 + 0.0im, nx, ny, nz)
    T2s = fill(60.0, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        B0[i,j,k] = Gx*xs[i] + Gy*ys[j]
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Smooth-random Gaussian field with complex S0 phase:
"""
function smooth_random_phantom(nx, ny, nz; σ=nx/8, TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    S0  = zeros(ComplexF64, nx, ny, nz)
    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    for k in 1:nz
        W1 = randn(nx, ny)
        W2 = randn(nx, ny)
        IW1 = imfilter(W1, Kernel.gaussian(σ))
        IW2 = imfilter(W2, Kernel.gaussian(σ))
        n1 = (IW1 .- minimum(IW1)) ./ (maximum(IW1)-minimum(IW1))
        n2 = (IW2 .- minimum(IW2)) ./ (maximum(IW2)-minimum(IW2))
        for j in 1:ny, i in 1:nx
            amp    = 0.8 + 0.4*n1[i,j]
            S0[i,j,k] = amp + 0.0im
            T2s[i,j,k] = 30.0 + 50.0*n2[i,j]
            B0[i,j,k]  = 5.0 * ((n1[i,j] + n2[i,j]) / 2)
        end
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

"""
Shepp–Logan phantom with complex S0 phase:
"""
function shepp_logan_phantom(nx, ny, nz; TE0=0.0, φ0=0.0, φx=0.0, φy=0.0)
    ellipses = [
        ( 2.0,  .69, .92,   0.0,   0.0,   0.0),
        (-0.98,.6624,.8740, 0.0, -0.0184, 0.0),
        (-0.02, .1100,.3100, .22,   0.0, -18.0),
        (-0.02, .1600,.4100,-.22,   0.0,  18.0),
        ( 0.01, .2100,.2500, 0.0,  .35,   0.0),
        ( 0.01, .0460,.0460, 0.0,  .1,    0.0),
        ( 0.02, .0460,.0460, 0.0, -.1,    0.0),
        ( 0.01, .0460,.0230,-.08, -.605,  0.0),
        ( 0.01, .0230,.0230, 0.0, -.606,  0.0),
        ( 0.02, .0230,.0460, .06, -.605,  0.0)
    ]
    t2vals = [60.0, 80.0, 50.0, 50.0, 70.0, 90.0, 90.0, 40.0, 40.0, 30.0]
    xs = LinRange(-1.0, 1.0, nx)
    ys = LinRange(-1.0, 1.0, ny)
    S0  = zeros(ComplexF64, nx, ny, nz)
    T2s = zeros(Float64, nx, ny, nz)
    B0  = zeros(Float64, nx, ny, nz)
    d0, dx, dy, Bsin = 0.0, 5.0, 0.0, 10.0
    for k in 1:nz, j in 1:ny, i in 1:nx
        x, y = xs[i], ys[j]
        for idx in 1:length(ellipses)
            A,a,b,x0,y0,phi = ellipses[idx]
            phir = deg2rad(phi)
            xp =  (x-x0)*cos(phir) + (y-y0)*sin(phir)
            yp = -(x-x0)*sin(phir) + (y-y0)*cos(phir)
            if (xp/a)^2 + (yp/b)^2 <= 1.0
                S0[i,j,k]  += A + 0.0im
                T2s[i,j,k]  = t2vals[idx]
                B0[i,j,k]   = d0 + dx*x + dy*y + Bsin*sin(2π*x)
            end
        end
    end
    apply_phase!(S0, B0, xs, ys; φ0=φ0, φx=φx, φy=φy, TE0=TE0)
    return S0, T2s, B0
end

end # module
