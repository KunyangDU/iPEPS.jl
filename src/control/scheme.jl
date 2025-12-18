struct FullSU <: AbstractAlgorithm end
struct FastSU <: AbstractAlgorithm end
mutable struct DynamicSU <: AbstractAlgorithm 
    N::Int64
    count::Int64
    DynamicSU(N::Int64) = new(N,0)
end