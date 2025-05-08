using DqT2

# Configure Settings
combine_coils = true
use_fat_modulation = false

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

raw, kx, ky, kz, config, sens, timepoints = load_and_process_data(combine_coils)

if use_fat_modulation
    @info "Using Fat Modulation"
    fat_modulation = calculate_fat_modulation(time_since_last_rf)
end

x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

x .= forward_op_synthetic_data_test(config, 
@view(kx[:, :, :, :]),
@view(ky[:, :, :, :]),
timepoints,
fat_modulation=use_fat_modulation ? fat_modulation : nothing,
[nx, ny, nz],
combine_coils = combine_coils,
timepoint_window_size=timepoint_window_size,
sens = combine_coils ? sens : nothing,
);

comb = combine_coils ? "" : "_no_combine_coils"
fat_mod = use_fat_modulation ? "_fatmod" : ""

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/x_$timepoint_window_size$comb$fat_mod", ComplexF32.(x))