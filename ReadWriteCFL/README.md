# ReadWriteCFL.jl

[![pipeline status](https://git.icr.ac.uk/icr-mr-in-rt/readwritecfl.jl/badges/master/pipeline.svg)](https://git.icr.ac.uk/icr-mr-in-rt/readwritecfl.jl/commits/master) [![coverage report](https://git.icr.ac.uk/icr-mr-in-rt/readwritecfl.jl/badges/master/coverage.svg)](https://git.icr.ac.uk/icr-mr-in-rt/readwritecfl.jl/commits/master)

## Presentation

Functions to save to file and load Complex data (the exact julia format is `ComplexF32`). This format matches with the original formats used in Matlab and it should be able to inter-operate with it.
The writing function writes two files (`data.hdr` and `data.cfl`). The hdr file contains the dimensions of the data and the cfl file contains the binary data.

## Usage

```julia
using ReadWriteCFL

# Providing data from the correct format
carr = rand(ComplexF32, (3, 3))

# Writing the data to file
ReadWriteCFL.writecfl("data.cfl", carr)
ReadWriteCFL.writecfl("data2", carr)

# Reading back the data
ReadWriteCFL.readcfl("data")
ReadWriteCFL.readcfl("data2.cfl")
```

## Notes

  - The Julia implementation matches the Matlab one:
    
      + Saved dimensions include a padding to 5 five dimensions with 1 ie: Vector => 5, 1, 1, 1, 1
      + Reloaded variables have same dimensions as the original stored data 5, 1, 1, 1, 1 => 5, 1

  - Julia makes a difference of types between (5, 1, 1, 1) and (5, 1)
    
      + Current implementation drops useless "1" dimensions when reading the data back
      + The function is probably ill-defined if ones attempt to store and load a single complex value
