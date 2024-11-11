includet("load_demo_data.jl")
includet("demo_recon_2d.jl")

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("raw_000.data", use_float32=true, use_nom_kz=true);

@assert size(noise) ==     ( 19832 ,   8)   # noise measurement could be used for pre-whitening
@assert size(raw)   ==     (  536  ,   8    ,   8    ,  32  , 269 )
# acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

using FFTW
# perform FFT across slices:
I = sortperm(-kz[1, 1, :, 1]); # looks like we need to flip the sign to make this work with standard FFT implementations ... we use ifft to account for this minus ...
v = kz[1, 1, I, 1];
@assert all(v ./ v[18] .≈ -16:15) # v[18] is the first positive kz coordinate, so basically v[18] = delta kz = 1/FOV

# this is working on the raw data, so it should only be executed once:
raw = raw[:, :, :, I, :];   # sorting by kz ...
raw = ifftshift(raw, 4); # since even number of partitions, fftshift and ifftshift will do the same thing ...
ifft!(raw, 4);
raw .*= sqrt(size(raw, 4)); # maintaining the norm ...
raw = fftshift(raw, 4);

# we'll use the kx and ky trajectory from the k-space centre plane:
kx = kx[:, :, 1, :]
ky = ky[:, :, 1, :]
kz = nothing

# low resolution reconstruction of echo 1 for coil sensitivity estimation:
x = demo_recon_2d(config, 
    @view(kx[:, 1, :]),
    @view(ky[:, 1, :]),
    @view(raw[:, :, 1, :, :]),
    [64, 64]
);

#using ImageView # alternative to arrShow, but doesn't work with complex and CuArray data
#imshow(abs.(x))

using ReadWriteCFL
ReadWriteCFL.writecfl("lowres_img", ComplexF32.(x))

# full resolution for image reconstruction:
nx = 256
ny = 256
nz = 32

# run external tool to estimate coil sensitivities (and interpolate to full image resolution):
run(`../bart-0.9.00/bart fft -u 7 lowres_img lowres_ksp`)
run(`../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz lowres_ksp ksp_zerop`)
run(`../bart-0.9.00/bart ecalib -t 0.01 -m1 ksp_zerop sens`)

# load coil sensitivities into Julia
sens = ReadWriteCFL.readcfl("sens");

#######################################################################################################################

# full-scale reconstruction (can loop over echoes):

combine_coils = true
x = combine_coils ? Array{ComplexF64}(undef, nx, ny, nz, config["necho"]) : Array{ComplexF64}(undef, nx, ny, nz, config["nchan"], config["necho"]);

for (ie, xe) in zip(1:config["necho"], eachslice(x, dims=length(size(x))))

    xe .= demo_recon_2d(config, 
        @view(kx[:, ie, :, :]),
        @view(ky[:, ie, :, :]),
        @view(raw[:, :, ie, :, :]),
        [nx, ny],
        combine_coils = combine_coils,
        sens = combine_coils ? sens : nothing,
        use_dcf = false, # for some reason this seems to introduce artifacts into the image ...
    );

end

ReadWriteCFL.writecfl("x", ComplexF32.(x))
