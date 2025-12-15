mutable struct LGState{Lx,Ly} <: AbstractState
    Γ::Vector{AbstractTensorMap}
    λ::Vector{AbstractTensorMap}
    nnsites::Tuple
    nntable::Dict{Tuple,Tuple}
    nn2d::Dict{Tuple,AbstractDirection}
    λindex::Tuple
    pspace::ElementarySpace
    function LGState(Latt::AbstractLattice)
        return new{size(Latt)...}(Vector{AbstractTensorMap}(),Vector{AbstractTensorMap}(),(),Dict{Tuple,Tuple}(),Dict{Tuple,AbstractDirection}(),(),ℂ^1)
    end
end

Base.length(::LGState{Lx, Ly}) where {Lx,Ly} = Lx*Ly
Base.size(::LGState{Lx, Ly}) where {Lx,Ly} = (Lx,Ly)


Base.getindex(ψ::LGState, i::Int64) = ψ.Γ[i],map(x -> ψ.λ[x],ψ.λindex[i])
_λindex(ψ::LGState, i::Int64, nn2λ::Dict) = map(x -> _λindex(ψ,i,x,nn2λ), (RIGHT(),UP(),DOWN(),LEFT()))
_λindex(ψ::LGState, i::Int64, d::AbstractDirection, nn2λ::Dict) = nn2λ[((i,[0,0]),ψ.nntable[(i,[0,0]),d])]


function Base.replace!(ψ::LGState, Γ::AbstractTensorMap, i::Int64)
    ψ.Γ[i] = Γ
end
Base.replace!(ψ::LGState, λ::AbstractTensorMap, i::Int64, ::RIGHT) = (ψ.λ[ψ.λindex[i][1]] = λ)
Base.replace!(ψ::LGState, λ::AbstractTensorMap, i::Int64, ::UP) = (ψ.λ[ψ.λindex[i][2]] = λ)
Base.replace!(ψ::LGState, λ::AbstractTensorMap, i::Int64, ::DOWN) = (ψ.λ[ψ.λindex[i][3]] = λ)
Base.replace!(ψ::LGState, λ::AbstractTensorMap, i::Int64, ::LEFT) = (ψ.λ[ψ.λindex[i][4]] = λ)
Base.replace!(ψ::LGState, λs::Tuple, i::Int64, ds::Tuple = (RIGHT(),UP(),DOWN(),LEFT())) = map(x -> replace!(ψ,λs[x],i,ds[x]), 1:4)
