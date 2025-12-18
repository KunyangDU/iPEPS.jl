
mutable struct SimpleUpdate{Sch} <: AbstractAlgorithm
    scheme::AbstractAlgorithm
    trunc::TensorKit.TruncationScheme
    tol::Float64
    N::Int64
    τs::Vector{Float64}
    τ::Float64
    noise::Float64
    function SimpleUpdate(
        scheme::AbstractAlgorithm,
        trunc::TensorKit.TruncationScheme,
        tol::Float64,
        N::Int64,
        τs::Vector{Float64},
        τ::Float64,
        noise::Float64
    )
        return new{typeof(scheme)}(scheme,trunc,tol,N,τs,τ,noise)  
    end

    function SimpleUpdate(scheme::AbstractAlgorithm,algo::SimpleUpdate)
        return new{typeof(scheme)}(scheme,algo.trunc,algo.tol,algo.N,algo.τs,algo.τ,algo.noise)
    end

    function SimpleUpdate(trunc::TensorKit.TruncationScheme)
        return new{FullSU}(FullSU(),trunc,0.0,0,Float64[],0.0,0.0)
    end

end
