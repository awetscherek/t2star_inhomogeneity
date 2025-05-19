function _gen_phantom(phantom_gen_func, eval_no, nchan)
    S0, T2, B0 = phantom_gen_func(nx,ny,nz)

    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_t2", ComplexF32.(T2))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_s0", ComplexF32.(S0))
    ReadWriteCFL.writecfl("/mnt/f/Dominic/Data/Synthetic/2d/$(eval_no)_b0", ComplexF32.(B0))
end

function gen_phantom_1(eval_no,nchan)
    _gen_phantom(SyntheticPhantoms.circle_phantom, eval_no, nchan)
end

function gen_phantom_2(eval_no,nchan)
    _gen_phantom(SyntheticPhantoms.concentric_circles_phantom, eval_no, nchan)
end

function gen_phantom_3(eval_no,nchan)
    _gen_phantom(SyntheticPhantoms.checkerboard_phantom, eval_no, nchan)
end

function gen_phantom_4(eval_no,nchan)
    _gen_phantom(SyntheticPhantoms.radial_rings_phantom, eval_no, nchan)
end

function gen_phantom_5(eval_no,nchan)
    _gen_phantom(SyntheticPhantoms.linear_gradient_phantom, eval_no, nchan)
end

function gen_phantom_6(eval_no,nchan)
    _gen_phantom(SyntheticPhantoms.smooth_random_phantom, eval_no, nchan)
end

function gen_phantom_7(eval_no,nchan)
    _gen_phantom(SyntheticPhantoms.shepp_logan_phantom, eval_no, nchan)
end