includet("load_demo_data.jl")

# using LsqFit
using ReadWriteCFL
using Plots
# using Statistics
using LinearAlgebra
using DSP

nx = 256
ny = 256
nz = 32

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true)

combine_coils = true
use_dcf = true

x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);
Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

# mag_s0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
# ϕs0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
# s0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

echo_times = time_since_last_rf[268,:,1,1]

# println("Echo times: ", echo_times)

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""

x .= ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Recon/2d/x$comb$dcf")
Δb0 .= Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/delta_b0$comb$dcf"))
# ϕs0 = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/B0/2d/init_phase$comb$dcf"))

println("Data Loaded")

#---

function fit_S0_R2_4D(echo_times,
                          S_data,
                          Δb0)

    γ = 2 * π * 42.576e6

    nx, ny, nz, nt = size(S_data)
    @assert length(echo_times) == nt "times length must match 4th dim of S_data"

    # design matrix [1  t]
    X = hcat(ones(nt), echo_times)

    S0 = Array{ComplexF64}(undef, nx,ny,nz)
    R2 = Array{Float64}(undef, nx,ny,nz)

    @inbounds for ix in 1:nx, iy in 1:ny, iz in 1:nz
        # 1) extract the time‐series at this voxel
        y = @view S_data[ix,iy,iz, :]

        # 2) remove known phase evolution exp(i γ B0 t)
        y_corr = y .* exp.(-im * γ * Δb0[ix,iy,iz] .* echo_times)

        # 3) linearize: w = log|y_corr| + i * unwrap(arg y_corr)
        ph = unwrap(angle.(y_corr))
        w  = log.(abs.(y_corr)) .+ im .* ph

        # 4) solve [1 t] · [log S0; p] ≈ w
        p = X \ w

        # 5) recover S0 and T2*
        S0[ix,iy,iz] = exp(p[1])
        R2[ix,iy,iz] = -real(p[2])
    end

    return S0, R2
end

s0, r2 = fit_S0_R2_4D(echo_times, x, Δb0)

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/t2$comb$dcf", ComplexF32.(1.0 ./ r2))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Intermediate/2d/s0$comb$dcf", ComplexF32.(s0))