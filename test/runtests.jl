includet("../operators/forward_operator.jl")
includet("../operators/adjoint_operator.jl")
includet("utils.jl")
includet("../load_demo_data.jl")
includet("../image_recon/image_recon_2d.jl")

config, noise, raw, kx, ky, kz, time_since_last_rf = load_demo_data("/mnt/f/Dominic/Data/raw_000.data", use_float32=true, use_nom_kz=true);

@assert size(noise) == (19832, 8)   # noise measurement could be used for pre-whitening
@assert size(raw) == (536, 8, 8, 32, 269)
# acquisition order: inner => nkx => nchan => necho => nkz => nky => outer

using ReadWriteCFL
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

# full resolution for image reconstruction:
nx = 256
ny = 256
nz = 32 #number of slices

# Configure Settings
combine_coils = true
use_dcf = true
use_fat_modulation = false

@info "Combine coils - $combine_coils"

if combine_coils

    if !isfile("coil_sens/sens.cfl")
        @info "No coil sensitivies found - creating coil sensitivity estimation"

        # low resolution reconstruction of echo 1 for coil sensitivity estimation:
        x = image_recon_2d(config,
            @view(kx[:, 1, :]),
            @view(ky[:, 1, :]),
            @view(raw[:, :, 1, :, :]),
            [64, 64]
        )

        ReadWriteCFL.writecfl("coil_sens/lowres_img", ComplexF32.(x))

        # run external tool to estimate coil sensitivities (and interpolate to full image resolution):
        run(`../../bart-0.9.00/bart fft -u 7 coil_sens/lowres_img coil_sens/lowres_ksp`)
        run(`../../bart-0.9.00/bart resize -c 0 $nx 1 $ny 2 $nz coil_sens/lowres_ksp coil_sens/ksp_zerop`)
        run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 coil_sens/ksp_zerop coil_sens/sens`)
    end

    # load coil sensitivities into Julia
    sens = ReadWriteCFL.readcfl("coil_sens/sens")
end
#######################################################################################################################

#Precision of approximation of timepoints
# 1 - No approximation (NUFFT for every time point)
# nkx (536) - Echo time of each assumed to be the timepoint
timepoint_window_size = 536

timepoints = vec(time_since_last_rf)

@assert timepoint_window_size <= nkx "The timepoint window size cannot be larger than nkx"

# this preconditioner could help speed up convergence:
dcf = use_dcf ? abs.(-size(ky, 1)/2+0.5:size(ky, 1)/2) : 1.0
dcf = dcf ./ maximum(dcf)

c_d = combine_coils ? sens : [1.0] # this shouldn't make a copy of sens

# use only raw data from 1st echo (most signal), normalize non-uniform frequency on pixel size (FOV/n)
kx_d = reshape(permutedims(kx, [2 1 3 4]) * config["FOVx"] / nx * 2 * pi, :, nky)
ky_d = reshape(permutedims(ky, [2 1 3 4]) * config["FOVy"] / ny * 2 * pi, :, nky)

# and use only data from central k-space region:
selection = -pi .<= kx_d .< pi .&& -pi .<= ky_d .< pi

# this is the raw data from which we want to reconstruct the coil images
#(num_timepoints, ky, nz * nchan)

dcf_y = use_dcf ? reshape(sqrt.(dcf), 1, size(dcf, 1), 1, 1, 1) : dcf

y_d = reshape(ComplexF64.(permutedims(raw, [3 1 5 4 2])) .* dcf_y, config["necho"] * nkx, :, nz * config["nchan"])[selection, :]

dcf_d = use_dcf ? repeat(sqrt.(dcf), outer=(size(ky, 2), size(ky, 3)))[selection] : 1.0

num_total_timepoints = config["necho"] * nkx
num_timepoints = ceil(Int, num_total_timepoints / timepoint_window_size)

# plan NUFFTs:
plan1 = finufft_makeplan(1, dims, -1, nz * config["nchan"], tol)    # type 1 (adjoint transform)
plan2 = finufft_makeplan(2, dims, 1, nz * config["nchan"], tol)     # type 2 (forward transform)

r = Array{ComplexF64}(undef, size(y_d))

function flatten(X,Y)
    return vcat(vec(X), vec(Y))
end

function unflatten(X)
    N = length(X) ÷ 2
    return reshape(X[1:N], size(e_d)), reshape(X[N+1:end], size(s0_d))
end

# Initialise Operators with implicit values
function forward_operator(x)
    e, s0 = unflatten(x)
    r .= forward_operator_impl(plan2, e, s0, num_timepoints, num_total_timepoints, kx_d, ky_d, c_d, timepoints, selection,
    timepoint_window_size, fat_modulation)
    r .*= dcf_d
    r .-= y_d
    obj = 1/2 * sum(abs2, r)
    @info "obj = $obj"
    return obj
end

function adjoint_operator!(storage, x)
    e, s0 = unflatten(x)
    g_e, g_s0 = adjoint_operator_impl(plan1, r, e, s0, dcf_d, combine_coils, c_d, num_timepoints, num_total_timepoints,
    timepoints, kx_d, ky_d, selection, use_dcf, timepoint_window_size, fat_modulation, nx, ny, nz, config["nchan"])
    storage .= flatten(g_e, g_s0)
end

check_objective_gradient_consistency(
    operator::Operator{PT},
    forward_dimensions::Tuple{Vararg{Int}},
    rtol::Float32=0.05f0,
    epsilon::Float64=5e-2;
    repetitions::Int=30,
)