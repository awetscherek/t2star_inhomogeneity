using DqT2

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation)

t2_star_mapping = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
s0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
Δb0 = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);

t2_star_mapping, s0, Δb0 = t2_recon_synthetic_data_test(config, 
@view(kx[:, :, :, :]),
@view(ky[:, :, :, :]),
@view(raw[:, :, :, :, :]),
timepoints,
fat_modulation=use_fat_modulation ? fat_modulation : nothing,
[nx, ny],
combine_coils = combine_coils,
timepoint_window_size=timepoint_window_size,
sens = combine_coils ? sens : nothing,
use_dcf = use_dcf, # for some reason this seems to introduce artifacts into the image ...
);

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""
fat_mod = use_fat_modulation ? "_fatmod" : ""

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/t2_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(t2_star_mapping))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/s0_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(s0))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/b0_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(Δb0))