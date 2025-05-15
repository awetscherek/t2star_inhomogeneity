#Experiment 1
# Circle with T2 as 50 and 
# B0 is zero

function gen_synthetic_data_1()
    @info "Eval 1"
    # Dimensions
    nx, ny, nz = 256, 256, 32
    radius = 100.0
    center_x, center_y = (nx + 1) / 2, (ny + 1) / 2

    # Initialize mappings
    S0 = zeros(Float64, nx, ny, nz)
    T2star = zeros(Float64, nx, ny, nz)
    B0 = zeros(Float64, nx,ny,nz)

    # Create 2D mask for a circle
    circle_mask = [hypot(x - center_x, y - center_y) <= radius for x in 1:nx, y in 1:ny]

    # Fill in the cylinder values for each slice
    for z in 1:nz
        S0[:, :, z] .= circle_mask .* 1.0
        T2star[:, :, z] .= circle_mask .* 50.0  
    end

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/1_t2", ComplexF32.(T2star))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/1_s0", ComplexF32.(S0))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/1_b0", ComplexF32.(B0))
end