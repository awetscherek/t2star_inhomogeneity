using DqT2

use_fat_modulation = false

# Configure Settings
combine_coils = true
use_dcf = true
eval_no = 4
σ = nothing

gdmode = Lbfgs()

output_file = (gdmode isa Adam) ? "eval_results_adam.txt" : "eval_results_lbfgs.txt"

@assert eval_no >= 1 && eval_no <= 4

function evaluate(gt_t2, gt_s0, gt_b0, rc_t2, rc_s0, rc_b0)

    l2_t2 = rmse(gt_t2, rc_t2)
    l2_s0 = rmse(gt_s0, rc_s0)
    l2_b0 = rmse(gt_b0, rc_b0)
    l2_total = l2_t2 + l2_s0 + l2_b0

    info="T2 Loss: $l2_t2 \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    info="S0 Loss: $l2_s0 \n"
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

function rmse(gt, rc)
    diff = gt .- rc
    N = length(diff)
    return sqrt(sum(abs2, diff) / N)
end

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation, true)

y_d, intermediate_t2, intermediate_s0, intermediate_b0 = load_synthetic_data(eval_no, config, combine_coils, sens, kx, ky, use_dcf, timepoints, fat_modulation, σ)

open(output_file, "a") do f
    println(f, "\n \n Evaluation $eval_no with σ=$(isnothing(σ) ? 0 : σ):")
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

gt_t2 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2")
gt_s0 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0")
gt_b0 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0")

info="DQT2: \n Timepoint Window Size: $timepoint_window_size \n"
@info info
open(output_file, "a") do f
    println(f, string(info))
end
evaluate(gt_t2, gt_s0, gt_b0, t2, s0_water, Δb0)

info="Intermediate Image: \n"
@info info
open(output_file, "a") do f
    println(f, string(info))
end
evaluate(gt_t2, gt_s0, gt_b0, intermediate_t2, intermediate_s0, intermediate_b0)

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