function calculate_fat_modulation(timepoints)

    nkx, necho, _, _ = size(timepoints)

    fat_modulation = zeros(ComplexF64, necho, nkx)
    
    # Relative amplitudes and Frequency offsets using the six peak model
    frequency_offsets = Float64[-3.80, -3.40, -2.60, -1.94, -0.39, 0.60] #TODO: check scaling
    relative_amplitudes = Float64[0.087 0.693 0.128 0.004 0.039 0.048] 

    for ie in 1:necho
        
        #(nkx,)
        t_echo = @view timepoints[:,ie,1,1]

        @inbounds for (amp, freq) in zip(relative_amplitudes, frequency_offsets)
            fat_modulation[ie, :] .+= amp .* exp.(im .* 2 .* π .* freq .* t_echo)
        end
    end
    
    return fat_modulation
end