
mutable struct SimpleUpdate <: AbstractAlgorithm
    trunc::TensorKit.TruncationScheme
    tol::Float64
    N::Int64
    τs::Vector{Float64}
    τ::Float64
    noise::Float64
end
