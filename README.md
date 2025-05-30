# DqT2
by Dominic Qu

This package includes forward and jacobian operators for direct reconstruction of T2* in K-space, as well as an iterative algorithm utilising the operators.

The operators take fat modulation and off-resonance from magnetic field inhomogeneity into account.

## Related research

'Improving T2* Estimation for Detection of Tumour Hypoxia in an MR-Linac' by Dominic Qu

An Imperial College Department of Computing MEng Individual Project with the Institute of Cancer Research

Supervisors: Andreas Wetscherek, Wayne Luk

## Basic Julia workflow

Install Julia - preferably linux / WSL (bart doesn't really support Windows)
Install bart (Berkley Advanced Image Reconstruction Toolbox) - this is technically only used to estimate coil sensitivity profiles, but it is useful.

VSCode with Julia Extension (setting "Julia Path" determines julia version to be used if not on path)

Open Folder - Project.toml in that folder contains information about the Packages used by the project / package

Start REPL - Julia Extension allows for inspection of variables 

switch to current folder (`]` switches to pkg shell (blue), backspace to get back):
```julia
]activate .
```

When opening this folder for the first time need to add local package as a development dependency (read/write support for BART's cfl format)
```julia
]dev ./ReadWriteCFL
```

If you have access to an NVIDIA GPU with > 16GB, you could try out the cuda scripts, I didn't include the CUDA package as a dependency by default, so if you want to add it:
```julia
]add CUDA
``` 

## Viewing Reconstructions

The best viewer that I've come across for complex-valued multi-dimensional data is the MATLAB tool arrShow (https://github.com/tsumpf/arrShow, command `as`). BART contains readcfl.m and writecfl.m to load .cfl data into MATLAB and display it.

## Result Scripts

### Real Data

Real data can be run to generate T2*, S0 mappings

`scripts/T2Recon/RealData/script_T2star_2d_mapping.jl` - Reconstructs real data 

### Synthetic Data

#### Evaluation of k-Space

##### Test single Evaluation Experiment
`scripts/T2Recon/Synthetic/Ksp/script_ksp_eval_all_synthetic.jl`

##### Test All Evaluation Experiment across different window sizes
`scripts/T2Recon/Synthetic/Ksp/script_ksp_eval_synthetic.jl`

#### Evaluation of T2* Reconstructions

##### Test single Evaluation Experiment
`scripts/T2Recon/Synthetic/Iterative/script_T2star_eval_all_synthetic.jl`

##### Test All Evaluation Experiment across different window sizes
`scripts/T2Recon/Synthetic/Iterative/script_T2star_eval_synthetic.jl`
