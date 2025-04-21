includet("load_demo_data.jl")

using ReadWriteCFL
using DSP
using Polynomials

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic_Data/Data/raw_000.data", use_float32=true, use_nom_kz=true);

@assert size(noise) ==     ( 19832 ,   8)   # noise measurement could be used for pre-whitening
@assert size(raw)   ==     (  536  ,   8    ,   8    ,  32  , 269 )
# acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

combine_coils = true
use_dcf = true

nx = 256
ny = 256
nz = 32 #number of slices

x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);


comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""

time_since_last_rf = Float64.(vec(time_since_last_rf))

echo_times = time_since_last_rf[268,:,1,1]

println("echo times: $echo_times")

x .= ReadWriteCFL.readcfl("/mnt/f/Dominic_Data/Results/Recon/2d/x$comb$dcf")

phases = permutedims(angle.(x), [4 1 2 3])

println("size of phases")
println(size(phases))

unwrap!(phases, dims=1)

p = fit(echo_times, phases, 1)

println("p is")
println(p)

b0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);


ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/Results/B0/2d/b0_prediction", ComplexF32.(b0))