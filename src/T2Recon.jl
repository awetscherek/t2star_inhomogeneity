abstract type GDMode end
struct Adam <: GDMode end
struct Lbfgs <: GDMode end

include("T2Recon/recon_2d_T2star_map_adam.jl")
include("T2Recon/recon_2d_T2star_map_lbfgs.jl")
include("T2Recon/apply_forward_op.jl")