using DqT2

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false
eval_no = 1

gdmode = Adam() # Lbfgs()

output_file = (gdmode isa Adam) ? "eval_results_adam.txt" : "eval_results_lbfgs.txt"

@assert eval_no >= 1 && eval_no <= 7

function l2_norm(gt, rc)
    diff = gt .- rc
    return sqrt(sum(abs2,diff))
end

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation, true)

open(output_file, "a") do f
    println(f, "Evaluation $eval_no:")
end

t2, s0_fat, s0_water, Δb0 = recon_2d_t2star_map(config,
    @view(kx[:, :, :, :]),
    @view(ky[:, :, :, :]),
    @view(raw[:, :, :, :, :]),
    timepoints,
    fat_modulation=use_fat_modulation ? fat_modulation : nothing,
    [nx, ny],
    gdmode,
    combine_coils=combine_coils,
    timepoint_window_size=timepoint_window_size,
    sens=sens,
    use_dcf=use_dcf, # for some reason this seems to introduce artifacts into the image ...
    use_synthetic=true,
    eval_no = eval_no
);

ground_truth = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2")
intermediate_t2 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Synthetic/2d/InitialPrediction/$(eval_no)_t2")

dqt2_loss = l2_norm(ground_truth, t2)
intermediate_loss = l2_norm(ground_truth, intermediate_t2)

info="DQT2: \n Timepoint Window Size: $timepoint_window_size \n Loss: $dqt2_loss"
@info info
open(output_file, "a") do f
    println(f, string(info))
end

info="Intermediate Image: \n Loss: $intermediate_loss"
@info info
open(output_file, "a") do f
    println(f, string(info))
end

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""
fat_mod = use_fat_modulation ? "_fatmod" : ""
water = use_fat_modulation ? "_water" : ""

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_t2_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(t2))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0$(water)_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(s0_water))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_delta_b0_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(Δb0))

if use_fat_modulation
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0_fat_$timepoint_window_size$comb$dcf$fat_mod", ComplexF32.(s0_fat))
end