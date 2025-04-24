includet("load_demo_data.jl")
includet("forward_op_synthetic_data_test.jl")
includet("demo_recon_2d.jl")

config, noise, _, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true);

@assert size(noise) ==     ( 19832 ,   8)   # noise measurement could be used for pre-whitening
# acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

using ReadWriteCFL
using FFTW
# perform FFT across slices:
I = sortperm(-kz[1, 1, :, 1]); # looks like we need to flip the sign to make this work with standard FFT implementations ... we use ifft to account for this minus ...
v = kz[1, 1, I, 1];
@assert all(v ./ v[18] .≈ -16:15) # v[18] is the first positive kz coordinate, so basically v[18] = delta kz = 1/FOV

# we'll use the kx and ky trajectory from the k-space centre plane:
kx = kx[:, :, 1, :]
ky = ky[:, :, 1, :]
kz = nothing

# full resolution for image reconstruction:
nx = 256
ny = 256
nz = 32 #number of slices

# low resolution reconstruction of echo 1 for coil sensitivity estimation:
combine_coils = false
if combine_coils
    # x = demo_recon_2d(config, 
    #     @view(kx[:, 1, :]),
    #     @view(ky[:, 1, :]),
    #     @view(raw[:, :, 1, :, :]),
    #     [64, 64]
    # );

    # #using ImageView # alternative to arrShow, but doesn't work with complex and CuArray data
    # #imshow(abs.(x))

    # ReadWriteCFL.writecfl("lowres_img", ComplexF32.(x))

    # # run external tool to estimate coil sensitivities (and interpolate to full image resolution):
    # run(`../../bart-0.9.00/bart fft -u 7 lowres_img lowres_ksp`)
    # run(`../../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz lowres_ksp ksp_zerop`)
    # run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 ksp_zerop sens`)

    # # load coil sensitivities into Julia
    # sens = ReadWriteCFL.readcfl("sens");

    #Assume all coils to have equal weighting
end
#######################################################################################################################

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

# full-scale reconstruction (can loop over echoes):

t2_star_mapping = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
s0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
Δb0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

t2_star_mapping, s0, Δb0 = forward_op_synthetic_data_test(config, 
@view(kx[:, :, :, :]),
@view(ky[:, :, :, :]),
time_since_last_rf,
[nx, ny, nz],
combine_coils = combine_coils,
niter=10,
timepoint_window_size=timepoint_window_size,
sens = combine_coils ? sens : nothing,
use_dcf = false, # for some reason this seems to introduce artifacts into the image ...
);

comb = combine_coils ? "" : "_no_combine_coils"

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/t2_$timepoint_window_size$comb", ComplexF32.(t2_star_mapping))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/s0_$timepoint_window_size$comb", ComplexF32.(s0))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/b0_$timepoint_window_size$comb", ComplexF32.(Δb0))