using DqT2

# Configure Settings
combine_coils = true
#whether the gt and reconstructed kspace are multiplied by dcf
use_dcf = true

dcf_weighted = true

use_fat_modulation = false

gdmode = Adam() # Lbfgs()
# gdmode = Lbfgs()

output_file = (gdmode isa Adam) ? "eval_results_ksp_adam.txt" : "eval_results_ksp_lbfgs.txt"

function rmse(gt, rc)
    diff = gt .- rc
    N = length(diff)
    return sqrt(sum(abs2, diff) / N)
end

function wrmse(gt, rc, dcf)
    diff = gt .- rc
    mse_w = sum(abs2.(diff) .* dcf) / sum(dcf)
    return sqrt(mse_w)
end

function evaluate(gt_ksp, rc_ksp, dcf)

    if dcf_weighted
        rmse_loss = wrmse(gt_ksp, rc_ksp,dcf)
    else
        rmse_loss = rmse(gt_ksp, rc_ksp)
    end

    info="KSP RMSE: $rmse_loss \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end
end

timepoint_window_sizes = [536, 536 ÷ 3, 536 ÷ 5, 536 ÷ 7, 536 ÷ 9, 536 ÷ 41, 3]
# timepoint_window_sizes = [536, cld(536,3), cld(536, 5), cld(536,7), cld(536, 9), cld(536, 51), 3]

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation, true)

# evals = [5,4,3,2]
# evals = [2,3,5]
evals=[3]

for eval_no in evals
    local info, y_d, fm, fat, water, gt_t2, gt_fat, gt_water, gt_b0

    info="\n \n KSPACE Evaluation $eval_no, dcf-weighted: $dcf_weighted:"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    #No gaussian noise
    y_d = load_synthetic_data(eval_no, config, combine_coils, sens, kx, ky, use_dcf, timepoints, fat_modulation, nothing, only_ksp=true)

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

    for tws in timepoint_window_sizes
        local timed, rc_ksp

        timed = @timed apply_forward_op(gt_t2, gt_b0, gt_water, gt_fat,
            config,
            @view(kx[:, :, :, :]),
            @view(ky[:, :, :, :]),
            timepoints,
            [nx, ny],
            fat_modulation=use_fat_modulation ? fat_modulation : nothing,
            combine_coils=combine_coils,
            timepoint_window_size=tws,
            sens=sens,
            use_dcf=use_dcf,
        );

        rc_ksp, dcf = timed.value

        info="DQT2: \n Timepoint Window Size: $tws \n Runtime: $(timed.time) seconds \n"
        @info info
        open(output_file, "a") do f
            println(f, string(info))
        end

        evaluate(y_d, rc_ksp, dcf)
    end
end