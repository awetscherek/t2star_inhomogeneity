includet("load_demo_data.jl")

using ReadWriteCFL
using DSP
using Polynomials
using Statistics

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true);

@assert size(noise) ==     ( 19832 ,   8)   # noise measurement could be used for pre-whitening
@assert size(raw)   ==     (  536  ,   8    ,   8    ,  32  , 269 )
# acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

combine_coils = true
use_dcf = true

nx = 256
ny = 256
nz = 32 #number of slices

x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

γ = 2 * π * 42.576e6

time_since_last_rf = Float64.(time_since_last_rf)

echo_times = vec(time_since_last_rf[268,:,1,1])

println("echo times: $echo_times")

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""
x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$comb$dcf")

mag = mean(abs.(x), dims=4)   # → size (nx,ny,nz)
mask = dropdims(mag .> 1e-7, dims=4)   # boolean mask of in-object voxels

phases = permutedims(angle.(x), [4 1 2 3])

#Phases size - (echo, nx, ny, nz)
unwrap!(phases, dims=1)

#Unwrapped size - (echo, nx * ny * nz)
unwrapped = reshape(phases, size(phases)[1], :)

N  = nx * ny * nz

#phase[S(t)] = y * Δb0 * t + phase[S(0)]
# Fit to phase[S(t)] = a * t + b
# Where a = y * Δb0
# b = phase[S(0)]

a = Vector{Float64}(undef, N)
b = Vector{Float64}(undef, N)

for i in 1:N
    if !mask[i]
        a[i] = 0.0
        b[i] = 0.0
        continue
    end

    p = fit(echo_times, @view(unwrapped[:,i]), 1)
    c = coeffs(p)

    # pad with zeros
    if length(c) < 2
        resize!(c, 2)
    end

    a[i] = c[1]
    b[i] = c[2]
end

Δb0  = reshape(a ./ γ, nx, ny, nz)
init_phase = reshape(b, nx, ny, nz)

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$comb$dcf", ComplexF32.(Δb0))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/B0/2d/init_phase$comb$dcf", ComplexF32.(init_phase))