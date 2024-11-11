# Image Reconstruction Demo - multiecho radial

## Basic Julia workflow

Install Julia - preferably linux / WSL (bart doesn't really support Windows)
Install bart (Berkley Advanced Image Reconstruction Toolbox) - this is technically only used to estimate coil sensitivity profiles, but it is useful.

VSCode with Julia Extension (setting "Julia Path" determines julia version to be used if not on path)

Open Folder - Project.toml in that folder contains information about the Packages used by the project / package

Start REPL - Julia Extension allows for inspection of variables 

switch to current folder (`]` switches to pkg shell, backspace to get back):
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

The script uses the includet function from the Revise package (which usually is automatically loaded in the REPL, if it is a project dependency). To load it manually:
```julia
using Revise
```

The best viewer that I've come across for complex-valued multi-dimensional data is the MATLAB tool arrShow (https://github.com/tsumpf/arrShow, command `as`). BART contains readcfl.m and writecfl.m to load .cfl data into MATLAB and display it.

For this demo, I use a relatively simple reconstruction with just a normal gradient descent algorithm... (which works with additional regularization, too). A faster convergence could maybe be achieved with other algorithms, such as "minres".

For the relaxation model, `time_since_last_rf` is the time to consider in the signal evolution.

Have fun!
