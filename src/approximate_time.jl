function approximate_time(timepoints, t_start, t_end, method::Symbol = :nearest_neighbour)
    if method == :nearest_neighbour
        return nearest_neighbour(timepoints, t_start, t_end)
    # elseif method == :linear_interpolation
    #     return linear_interpolation(timepoints, t_start, t_end)
    else
        throw(ArgumentError("Unknown Method: $method"))
    end
end

function nearest_neighbour(timepoints, t_start, t_end)
    t_index = (t_start + t_end) ÷ 2

    te_index = contains_te(t_start,t_end)
    if !isnothing(te_index)
        return timepoints[268 + te_index * 536]
    end

    return timepoints[t_index]
end

function contains_te(t_start, t_end)
    # lowest integer m that makes 268 + 536*m ≥ start
    m_min = ceil(Int,    (t_start  - 268) / 536)
    # highest integer m that makes 268 + 536*m ≤ finish
    m_max = floor(Int,   (t_end - 268) / 536)
    return m_min ≤ m_max ? m_min : nothing
end
