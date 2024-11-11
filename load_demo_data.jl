using Mmap, MAT

function load_demo_data(filename; use_float32 = false, use_nom_kz = false)

    # these parameters are the same for all data sets:
    config = Dict(
        "nkx" => 536,                   # samples per line
        "nchan" => 8,                   # number of receiver channels
        "necho" => 8,                   # number of echo times
        "nky" => 269,                   # number of radial projections
        "nkz" =>  32,                   # number of k-space partitions (discrete kz values)
        "offset" => 1269248,            # position in file where raw data starts (noise adjustment before)
        "FOVx" => 0.400,                # Acquisition FOV in meters
        "FOVy" => 0.400,
        "FOVz" => 0.128,
    )

    noise, raw = open(filename) do io 
        mmap(io, Array{ComplexF32, 2}, (config["offset"] ÷ (8 * config["nchan"]), config["nchan"])),
        mmap(io, Array{ComplexF32, 5}, tuple([config[ax] for ax in ["nkx", "nchan", "necho", "nkz", "nky"]]...), config["offset"]);
    end

    noise64 = ComplexF64.(noise)
    raw64 = ComplexF64.(raw)

    kx, ky, kz, b0, time_since_last_rf = matopen("traj_cor.mat", "r") do io
        [read(io, name) for name in ["kx", "ky", "kz", "b0", "time_since_last_rf"]]
    end

    # Now all the data is loaded ...
    for c = 1:config["nchan"]
        # apply coil-specific phase correction:
        raw64[:, c, :, :, :] .*= exp.(b0[c] * (-42.576 * 2 * pi * 1im))
        # and perform a half-FOV shift by multiplying with a phase ramp:
        raw64[:, c, :, :, :] .*= exp.(kz * (config["FOVz"] * pi * 1im))
    end

    if use_nom_kz
        kz = matopen(io->read(io, "kz"), "traj_nom.mat", "r")
    end

    if use_float32
        config, ComplexF32.(noise64), ComplexF32.(raw64), Float32.(kx), Float32.(ky), Float32.(kz), Float32.(time_since_last_rf)
    else
        config, noise64, raw64, kx, ky, kz, time_since_last_rf
    end
end

