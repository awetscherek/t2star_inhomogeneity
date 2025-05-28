using DqT2

use_fat_modulation = true

# Configure Settings
combine_coils = true
use_dcf = true
eval_no = 1

gdmode = Adam()

output_file = (gdmode isa Adam) ? "eval_results_fatmod_adam.txt" : "eval_results_fatmod_lbfgs.txt"

@assert eval_no >= 1 && eval_no <= 2

function evaluate_dqt2(gt_t2, gt_fat, gt_water, gt_b0, rc_t2, rc_fat, rc_water, rc_b0)
        
    l2_t2 = l2_norm(gt_t2 ./ 1000, rc_t2 ./ 1000)
    l2_fat = l2_norm(gt_fat, rc_fat)
    l2_water = l2_norm(gt_water, rc_water)
    l2_b0 = l2_norm(gt_b0, rc_b0)
    l2_total = l2_t2 + l2_fat + l2_water + l2_b0

    info="T2 Loss: $l2_t2 \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    info="Fat Loss: $l2_fat \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    info="Water Loss: $l2_water \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    info="B0 Loss: $l2_b0 \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    info="Total Loss: $l2_total \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end
end

function evaluate_intermediate(gt_t2, gt_b0, rc_t2, rc_b0)
        
    l2_t2 = l2_norm(gt_t2 ./ 1000, rc_t2 ./ 1000)
    l2_b0 = l2_norm(gt_b0, rc_b0)

    info="T2 Loss: $l2_t2 \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    info="B0 Loss: $l2_b0 \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end
end

function l2_norm(gt, rc)
    diff = gt .- rc
    return sqrt(sum(abs2,diff))
end

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation, true)

y_d, intermediate_t2, intermediate_s0, intermediate_b0 = load_synthetic_data_fatmod(eval_no, config, combine_coils, sens, kx, ky, use_dcf, timepoints, fat_modulation)

open(output_file, "a") do f
    println(f, "\n \n Evaluation $eval_no:")
end

t2, s0_fat, s0_water, Δb0 = recon_2d_t2star_map(config,
    @view(kx[:, :, :, :]),
    @view(ky[:, :, :, :]),
    y_d,
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

gt_t2 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2_fatmod")
gt_fat = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_fat")
gt_water = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water")
gt_b0 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0_fatmod")

info="DQT2: \n Timepoint Window Size: $timepoint_window_size \n"
@info info
open(output_file, "a") do f
    println(f, string(info))
end
evaluate_dqt2(gt_t2, gt_fat, gt_water, gt_b0, t2, s0_fat, s0_water, Δb0)

info="Intermediate Image: \n"
@info info
open(output_file, "a") do f
    println(f, string(info))
end
evaluate_intermediate(gt_t2, gt_b0, intermediate_t2, intermediate_b0)

comb = combine_coils ? "" : "_no_combine_coils"
dcf = use_dcf ? "_dcf" : ""
fat_mod = use_fat_modulation ? "_fatmod" : ""
water = use_fat_modulation ? "_water" : ""
mode = (gdmode isa Adam) ? "_adam" : "_lbfgs"

ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_t2_$timepoint_window_size$comb$dcf$mode$fat_mod", ComplexF32.(t2))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0$(water)_$timepoint_window_size$comb$dcf$mode$fat_mod", ComplexF32.(s0_water))
ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_delta_b0_$timepoint_window_size$comb$dcf$mode$fat_mod", ComplexF32.(Δb0))

if use_fat_modulation
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0_fat_$timepoint_window_size$comb$dcf$mode$fat_mod", ComplexF32.(s0_fat))
end