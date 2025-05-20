include("InitialPredictions/RealData/real_b0_prediction.jl")
include("InitialPredictions/Synthetic/synthetic_b0_prediction.jl")

include("InitialPredictions/gen_s0.jl")
include("InitialPredictions/RealData/real_s0_prediction.jl")
include("InitialPredictions/Synthetic/generate_intermediate_image_prediction.jl")

include("InitialPredictions/RealData/initialise_params.jl")
include("InitialPredictions/Synthetic/initialise_params.jl")

include("InitialPredictions/initial_predictions.jl")