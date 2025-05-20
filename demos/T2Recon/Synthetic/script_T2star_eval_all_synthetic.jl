using DqT2

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false

gdmode = Adam() # Lbfgs()
# gdmode = Lbfgs()

output_file = (gdmode isa Adam) ? "eval_results_adam.txt" : "eval_results_lbfgs.txt"

function l2_norm(gt, rc)
    diff = gt .- rc
    return sqrt(sum(abs2,diff))
end

timepoint_window_sizes = [536, 268, 134, 67, 30, 20]

raw, kx, ky, kz, config, sens, timepoints, fat_modulation = load_and_process_data(combine_coils, use_fat_modulation, true)

for eval_no in 1:7

    info="\n \n Evaluation $eval_no:"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    y_d, intermediate_t2 = load_synthetic_data(eval_no, config, combine_coils, sens, kx, ky, use_dcf, fat_modulation)

    ground_truth = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2")

    intermediate_loss = l2_norm(ground_truth, intermediate_t2)
    info="Intermediate Image: \n Loss: $intermediate_loss"
    @info info
    open(output_file, "a") do f
        println(f, string(info))
    end

    for tws in timepoint_window_sizes
        t2, s0_fat, s0_water, Δb0 = recon_2d_t2star_map(config,
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
            use_dcf=use_dcf, # for some reason this seems to introduce artifacts into the image ...
            use_synthetic=true,
            eval_no = eval_no
        );

        comb = combine_coils ? "" : "_no_combine_coils"
        dcf = use_dcf ? "_dcf" : ""
        fat_mod = use_fat_modulation ? "_fatmod" : ""
        water = use_fat_modulation ? "_water" : ""

        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_t2_$tws$comb$dcf$fat_mod", ComplexF32.(t2))
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0$(water)_$tws$comb$dcf$fat_mod", ComplexF32.(s0_water))
        ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_delta_b0_$tws$comb$dcf$fat_mod", ComplexF32.(Δb0))

        if use_fat_modulation
            ReadWriteCFL.writecfl("/mnt/f/Dominic/Results/Synthetic/2d/Results/$(eval_no)_s0_fat_$tws$comb$dcf$fat_mod", ComplexF32.(s0_fat))
        end

        dqt2_loss = l2_norm(ground_truth, t2)

        info="DQT2: \n Timepoint Window Size: $tws \n Loss: $dqt2_loss"
        @info info
        open(output_file, "a") do f
            println(f, string(info))
        end
    end
end