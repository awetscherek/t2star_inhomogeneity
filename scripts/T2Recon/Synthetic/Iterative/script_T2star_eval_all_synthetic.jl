using DqT2

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false
σ = nothing

gdmode = Adam() # Lbfgs()
# gdmode = Lbfgs()

output_file = (gdmode isa Adam) ? "eval_results_adam.txt" : "eval_results_lbfgs.txt"

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

timepoint_window_sizes = [536, 536 ÷ 3, 536 ÷ 5, 536 ÷ 7, 536 ÷ 9, 536 ÷ 41]

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation, true)

evals = [5,3,2]

for eval_no in evals
    local info, y_d, intermediate_t2, intermediate_s0, intermediate_b0, gt_t2, gt_s0, gt_b0

    info="\n \n Evaluation $eval_no with σ=$(isnothing(σ) ? 0 : σ):"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    y_d, intermediate_t2, intermediate_s0, intermediate_b0 = load_synthetic_data(eval_no, config, combine_coils, sens, kx, ky, use_dcf, timepoints, fat_modulation, σ)

    gt_t2 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2")
    gt_s0 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0")
    gt_b0 = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0")

    info="Intermediate Image: \n"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end
    evaluate(gt_t2, gt_s0, gt_b0, intermediate_t2, intermediate_s0, intermediate_b0)

    for tws in timepoint_window_sizes
        local timed, t2, s0_fat, s0_water, Δb0, comb, dcf, fat_mod, water, mode

        timed = @timed recon_2d_t2star_map(config,
            @view(kx[:, :, :, :]),
            @view(ky[:, :, :, :]),
            y_d,
            timepoints,
            fat_modulation=use_fat_modulation ? fat_modulation : nothing,
            [nx, ny],
            gdmode,
            combine_coils=combine_coils,
            timepoint_window_size=tws,
            sens=sens,
            use_dcf=use_dcf,
            use_synthetic=true,
            eval_no = eval_no,
            σ=σ
        );

        t2, s0_fat, s0_water, Δb0 = timed.value

        comb = combine_coils ? "" : "_no_combine_coils"
        dcf = use_dcf ? "_dcf" : ""
        fat_mod = use_fat_modulation ? "_fatmod" : ""
        water = use_fat_modulation ? "_water" : ""
        mode = (gdmode isa Adam) ? "_adam" : "_lbfgs"

        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_t2_$tws$comb$dcf$mode$fat_mod", ComplexF32.(t2))
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0$(water)_$tws$comb$dcf$mode$fat_mod", ComplexF32.(s0_water))
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_delta_b0_$tws$comb$dcf$mode$fat_mod", ComplexF32.(Δb0))

        if use_fat_modulation
            ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0_fat_$tws$comb$dcf$mode$fat_mod", ComplexF32.(s0_fat))
        end

        info="DQT2: \n Timepoint Window Size: $tws \n Runtime: $(timed.time) seconds \n"
        @info info
        open(output_file, "a") do f
            println(f, string(info))
        end
        evaluate(gt_t2, gt_s0, gt_b0, t2, s0_water, Δb0)
    end
end