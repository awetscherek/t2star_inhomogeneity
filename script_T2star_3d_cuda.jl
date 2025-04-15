includet("load_demo_data.jl")
includet("demo_recon_3d_cuda.jl")

using ReadWriteCFL

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic_Data/raw_000.data", use_float32=true);

@assert size(noise) ==     ( 19832 ,   8)   # noise measurement could be used for pre-whitening
@assert size(raw)   ==     (  536  ,   8    ,   8    ,  32  , 269 )
# acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

# full resolution for image reconstruction:
nx = 256
ny = 256
nz = 32

# low resolution reconstruction of echo 1 for coil sensitivity estimation:
combine_coils = false
if combine_coils
    x = demo_recon_3d(config, 
        @view(kx[:, 1, :, :]),
        @view(ky[:, 1, :, :]),
        @view(kz[:, 1, :, :]),
        @view(raw[:, :, 1, :, :]),
        [64, 64, 16]
    );

    #using ImageView # alternative to arrShow, but doesn't work with complex and CuArray data
    #imshow(abs.(x))

    ReadWriteCFL.writecfl("lowres_img", ComplexF32.(x))

    # run external tool to estimate coil sensitivities (and interpolate to full image resolution):
    run(`../../bart-0.9.00/bart fft -u 7 lowres_img lowres_ksp`)
    run(`../../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz lowres_ksp ksp_zerop`)
    run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 ksp_zerop sens`)

    # load coil sensitivities into Julia
    sens = ReadWriteCFL.readcfl("sens");
end
#######################################################################################################################

# full-scale reconstruction (can loop over echoes):
x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

for (ie, xe) in zip(1:config["necho"], eachslice(x, dims=length(size(x))))

    xe .= demo_recon_3d(config, 
        @view(kx[:, ie, :, :]),
        @view(ky[:, ie, :, :]),
        @view(kz[:, ie, :, :]),
        @view(raw[:, :, ie, :, :]),
        [nx, ny, nz],
        combine_coils = combine_coils,
        sens = combine_coils ? sens : nothing,
        use_dcf = true, # for some reason this seems to introduce artifacts into the image, so it might be required to add some regularisation ...
    );

end

ReadWriteCFL.writecfl("/mnt/f/Dominic_Data/Results/Demo/x_3d", ComplexF32.(x))
