includet("../load_demo_data.jl")
includet("recon_2d_T2star_map.jl")
includet("../image_recon/image_recon_2d.jl")
includet("../fat_modulation.jl")

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true);

@assert size(noise) == (19832, 8)   # noise measurement could be used for pre-whitening
@assert size(raw) == (536, 8, 8, 32, 269)
# acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

using ReadWriteCFL
using FFTW
# perform FFT across slices:
I = sortperm(-kz[1, 1, :, 1]); # looks like we need to flip the sign to make this work with standard FFT implementations ... we use ifft to account for this minus ...
v = kz[1, 1, I, 1];
@assert all(v ./ v[18] .≈ -16:15) # v[18] is the first positive kz coordinate, so basically v[18] = delta kz = 1/FOV

# this is working on the raw data, so it should only be executed once:
raw = raw[:, :, :, I, :];   # sorting by kz ...
raw = ifftshift(raw, 4); # since even number of partitions, fftshift and ifftshift will do the same thing ...
ifft!(raw, 4);
raw .*= sqrt(size(raw, 4)); # maintaining the norm ...
raw = fftshift(raw, 4);

# we'll use the kx and ky trajectory from the k-space centre plane:
kx = kx[:, :, 1, :]
ky = ky[:, :, 1, :]
kz = nothing

# full resolution for image reconstruction:
nx = 256
ny = 256
nz = 32 #number of slices

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false

@info "Combine coils - $combine_coils"

if combine_coils

    if !isfile("coil_sens/sens.cfl")
        @info "No coil sensitivies found - creating coil sensitivity estimation"

        # low resolution reconstruction of echo 1 for coil sensitivity estimation:
        x = image_recon_2d(config,
            @view(kx[:, 1, :]),
            @view(ky[:, 1, :]),
            @view(raw[:, :, 1, :, :]),
            [64, 64]
        )

        ReadWriteCFL.writecfl("coil_sens/lowres_img", ComplexF32.(x))

        # run external tool to estimate coil sensitivities (and interpolate to full image resolution):
        run(`../../bart-0.9.00/bart fft -u 7 coil_sens/lowres_img coil_sens/lowres_ksp`)
        run(`../../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz coil_sens/lowres_ksp coil_sens/ksp_zerop`)
        run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 coil_sens/ksp_zerop coil_sens/sens`)
    end

    # load coil sensitivities into Julia
    sens = ReadWriteCFL.readcfl("coil_sens/sens")
end
#######################################################################################################################

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

timepoints = vec(time_since_last_rf)

if use_fat_modulation
    fat_modulation = calculate_fat_modulation(time_since_last_rf)
end

# full-scale reconstruction (can loop over echoes):

t2_star_mapping = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
s0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
Δb0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

t2_star_mapping, s0, Δb0 = recon_2d_t2star_map(config,
    @view(kx[:, :, :, :]),
    @view(ky[:, :, :, :]),
    @view(raw[:, :, :, :, :]),
    timepoints,
    fat_modulation=use_fat_modulation ? fat_modulation : nothing,
    [nx, ny],
    combine_coils=combine_coils,
    timepoint_window_size=timepoint_window_size,
    sens=combine_coils ? sens : nothing,
    use_dcf=use_dcf, # for some reason this seems to introduce artifacts into the image ...
);

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""
fat_mod = use_fat_modulation ? "_fatmod" : ""

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/T2/2d/t2_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(t2_star_mapping))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/T2/2d/s0_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(s0))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/T2/2d/delta_b0_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(Δb0))