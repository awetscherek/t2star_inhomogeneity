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
    t_index = round(Int, (t_start + t_end) / 2)
    return timepoints[t_index]
end

function linear_interpolation(timepoints, t_start, t_end)
end