using ReadWriteCFL
using FFTW

# in_a = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/coil_sens/Synthetic/lowres_img")

# out = fft(in_a, (1,2,3),)

# # compute the total number of points transformed:
# N = prod(size(in_a, d) for d in (1,2,3))

# # scale to make it unitary (i.e. divide by sqrt(N)):
# out .*= 1/sqrt(N)

# ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/coil_sens/Synthetic/lowres_ksp", ComplexF32.(out))
# # run(`../../bart-0.9.00/bart ecalib -t 0.01 -m1 /mnt/f/Dominic/Data/coil_sens/Synthetic/ksp_zerop /mnt/f/Dominic/Data/coil_sens/Synthetic/1_sens`)


# run(`../../bart-0.9.00/bart fft -u 7 /mnt/f/Dominic/Data/coil_sens/Synthetic/coil_img /mnt/f/Dominic/Data/coil_sens/Synthetic/ksp_zerop`)
run(`../../bart-0.9.00/bart ecalib /mnt/f/Dominic/Data/coil_sens/Synthetic/lowres_ksp /mnt/f/Dominic/Data/coil_sens/Synthetic/1_sens`)


# in_a = ReadWriteCFL.readcfl("/mnt/f/Dominic/Data/coil_sens/Synthetic/lowres_ksp")

# println(maximum(abs.(in_a)))


