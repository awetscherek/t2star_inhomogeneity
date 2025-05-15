using DqT2

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false
eval_no = 1

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation)

t2 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);
s0_fat = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
s0_water = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"]);
Δb0 = combine_coils ? Array{Float64}(undef, nx, ny, nz) : Array{Float64}(undef, nx, ny, nz, config["nchan"]);

t2, s0_fat, s0_water, Δb0 = recon_2d_t2star_map(config,
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
    use_synthetic=true,
    eval_no = eval_no
);

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""
fat_mod = use_fat_modulation ? "_fatmod" : ""
water = use_fat_modulation ? "_water" : ""

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/$(eval_no)_t2_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(t2))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/$(eval_no)_s0$(water)_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(s0_water))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/$(eval_no)_delta_b0_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(Δb0))

if use_fat_modulation
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/$(eval_no)_s0_fat_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(s0_fat))
end