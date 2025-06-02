using ReadWriteCFL

function rmse(gt, rc)
    diff = gt .- rc
    N = length(diff)
    return sqrt(sum(abs2, diff) / N)
end

gt = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/Intermediate/2d/t2_dcf"))
rc = Float64.(ReadWriteCFL.readcfl("/mnt/f/Dominic/Results/T2/2d/t2_536_dcf_adam"))

rc[rc .== 50.0] .= 0.0

loss = rmse(gt,rc)

info = "Real data \n Evaluation between image-space reconstruction method and k-space reconstruction method \n Loss:$loss"
@info info