module DqT2

using CUDA
using FINUFFT
using Optimisers
using ProgressBars
using Optim
using ReadWriteCFL
using DSP
using Polynomials
using Statistics
using Plots
using LinearAlgebra
using FFTW
using Mmap
using MAT

export ReadWriteCFL

const γ = 2 * π * 42.576e6

# Resolutions for image reconstruction. 
# Constants as these values have been used throughout the project
const nx = 256
const ny = 256
const nz = 32 #number of slices


export nx,ny,nz

include("fat_modulation.jl")

include("Utils/Utils.jl")

export load_and_process_data, preprocess_data

include("ImageRecon.jl")

export image_recon_2d
export image_recon_3d

include("Operators.jl")

export forward_operator_impl
export adjoint_operator_impl

include("InitialPredictions.jl")

include("Synthetic.jl")

export SyntheticPhantoms
export load_synthetic_data, load_synthetic_data_fatmod

include("T2Recon.jl")

export recon_2d_t2star_map
export apply_forward_op
export GDMode, Adam, Lbfgs

end