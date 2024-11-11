module ReadWriteCFL

#=
Conversion from the WriteCFL and ReadCFL functions to
read and Write complex data to file.
Writes reconstruction data to filenamebase.cfl (complex float) and its
dimensions to filenamebase.hdr.

Initially written to edit data with the Berkeley Advanced Reconstruction Toolbox (BART).

Original file Copyright
% Copyright 2013. Joseph Y Cheng.
% Copyright 2016. CBClab, Maastricht University.
% 2012 Joseph Y Cheng (jycheng@mrsrl.stanford.edu).
% 2016 Tim Loderhose (t.loderhose@student.maastrichtuniversity.nl).

Adapted in julia by Bastien Lecoeur (bastien.lecoeur@icr.ac.uk) in 2021
=#

export writecfl, readcfl

"""
    remove_exe(file_path::String)

remove the extension from the file path by spliting into a pair of 'filename'
and 'ext' containing the directory and the extension of the file path and
returning only the filename.
"""
function remove_exe(file_path::String)
    # split extension from rest of path
    filename, _ = splitext(file_path)
    return filename
end

"""
    writecfl(filename::String, data::Array{ComplexF32})

Write the given `data` into a pair of `.cfl` and `.hdr` files
using `filename` as the base name.
"""
function writecfl(filename::String, data::Array{ComplexF32})
    # Write the dimension in the associated ".hdr" files
    filenamebase = remove_exe(filename)
    writehdr(filenamebase, size(data))

    # Write the data themselves as binary
    open(filenamebase * ".cfl", "w") do file
        return write(file, data)
    end
    return nothing
end

"""
    writehdr(filename::String, dims::Tuple)

Write the `.hdr` file containing the dimensions of the binary output file.
"""
@inline function writehdr(filename::String, dims::Tuple)
    filenamebase = remove_exe(filename)
    open(filenamebase * ".hdr", "w") do file
        write(file, "# Dimensions\n")
        for dim in dims
            write(file, string(dim) * " ")
        end
        if length(dims) < 5
            write(file, ("1 "^(5 - length(dims))) * "\n")
        end
    end
    return nothing
end

"""
    readcfl(filename::String)::Array{ComplexF32}

Read a pair of `.cfl` and `.hdr` file based on the `filenamebase` into a
Complex array of the correct dimensions. The array size is precised into
the `.hdr` file.
"""
function readcfl(filename::String)::Array{ComplexF32}
    filenamebase = remove_exe(filename)
    # Read the dimensions back
    dims::Tuple = readhdr(filenamebase)

    # Remove useless dimensions from the tuple 
    dims = dims[1:maximum(findall(x -> x != one(typeof(x)), dims))]
    rawdata = Array{ComplexF32}(undef, dims)
    # Read all the floats
    read!(filenamebase * ".cfl", rawdata)
    return rawdata
end

"""
    readhdr(filenamebase::String)::Tuple

Read a `.hdr` file containing the size of the `.cfl` file into a tuple.
"""
@inline function readhdr(filename::String)::Tuple
    filenamebase = remove_exe(filename)
    open(filenamebase * ".hdr", "r") do file
        # Jump initial comments
        line = readline(file)
        while line[1] == '#'
            line = readline(file)
        end

        # Parse the line containing the dimensions
        words::Vector{String} = split(line)
        dims::Vector{Int64} = map(x -> parse(Int64, x), words)
        return Tuple(dims)
    end
end

end # module