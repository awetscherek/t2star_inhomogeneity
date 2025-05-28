function _gen_phantom(phantom_gen_func, eval_no)
    S0, T2, B0 = phantom_gen_func(nx,ny,nz)

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2", ComplexF32.(T2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0", ComplexF32.(S0))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0", ComplexF32.(B0))
end

function gen_phantom_1(eval_no)
    _gen_phantom(SyntheticPhantoms.circle_phantom, eval_no)
end

function gen_phantom_2(eval_no)
    _gen_phantom(SyntheticPhantoms.checkerboard_phantom, eval_no)
end

function gen_phantom_3(eval_no)
    _gen_phantom(SyntheticPhantoms.linear_gradient_phantom, eval_no)
end

function gen_phantom_4(eval_no)
    _gen_phantom(SyntheticPhantoms.linear_gradient_t2_b0, eval_no)
end

#Fatmod

function _gen_phantom_fatmod(phantom_gen_func, eval_no)
    fat, water, T2, B0 = phantom_gen_func(nx,ny,nz)

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2_fatmod", ComplexF32.(T2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_fat", ComplexF32.(fat))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0_water", ComplexF32.(water))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0_fatmod", ComplexF32.(B0))
end

function gen_phantom_fatmod_1(eval_no)
    _gen_phantom_fatmod(SyntheticPhantoms.circle_phantom_fatmod, eval_no)
end

function gen_phantom_fatmod_2(eval_no)
    _gen_phantom_fatmod(SyntheticPhantoms.checkerboard_phantom_fatmod, eval_no)
end