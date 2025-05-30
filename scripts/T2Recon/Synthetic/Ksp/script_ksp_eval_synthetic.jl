using DqT2

use_fat_modulation = false

# Configure Settings
combine_coils = true
#whether the gt and reconstructed kspace are multiplied by dcf
use_dcf = true
eval_no = 5

gdmode = Adam()

output_file = (gdmode isa Adam) ? "eval_results_ksp_adam.txt" : "eval_results_ksp_lbfgs.txt"

@assert eval_no >= 1 && eval_no <= 5

function rmse(gt, rc)
    diff = gt .- rc
    N = length(diff)
    return sqrt(sum(abs2, diff) / N)
end

function evaluate(gt_ksp, rc_ksp)

    rmse_loss = rmse(gt_ksp, rc_ksp)

    info="KSP RMSE: $rmse_loss \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end
end

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536 ÷ 41

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation, true)

#Take σ to be nothing
y_d = load_synthetic_data(eval_no, config, combine_coils, sens, kx, ky, false, timepoints, fat_modulation, nothing, only_ksp=true)

open(output_file, "a") do f
    println(f, "\n \n KSPACE Evaluation $eval_no")
end

fm = use_fat_modulation ? "_fatmod" : ""
fat = use_fat_modulation ? "_fat" : ""
water = use_fat_modulation ? "_water" : ""

gt_t2 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2$fm")
gt_water = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0$water")
gt_b0 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0$fm")

if use_fat_modulation
    gt_fat = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0$fat")
else
    gt_fat = nothing
end

timed = @timed apply_forward_op(gt_t2, gt_b0, gt_water, gt_fat,
    config,
    @view(kx[:, :, :, :]),
    @view(ky[:, :, :, :]),
    timepoints,
    [nx, ny],
    fat_modulation=use_fat_modulation ? fat_modulation : nothing,
    combine_coils=combine_coils,
    timepoint_window_size=timepoint_window_size,
    sens=sens,
    use_dcf=use_dcf,
);

rc_ksp = timed.value

info="DQT2: \n Timepoint Window Size: $timepoint_window_size \n Runtime: $(timed.time) seconds \n"
@info info
open(output_file, "a") do f
    println(f, string(info))
end

evaluate(y_d, rc_ksp)