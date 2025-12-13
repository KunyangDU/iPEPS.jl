using TensorKit
include("../src/iPEPS.jl")

abstract type AbstractOperator end
mutable struct Hamiltonian <: AbstractOperator
    H2::Dict{Tuple,Matrix}
    H1::Dict{Int64,Vector}
    nnnpath::Union{Nothing,Dict{Tuple,Tuple}}
    function Hamiltonian(H2::Dict{Tuple,Matrix},
    H1::Dict{Int64,Vector})
        return new(H2,H1,nothing)
    end
    Hamiltonian() = new(Dict{Tuple,Matrix}(),Dict{Int64,Vector}(),nothing)
end

abstract type AbstractState end
mutable struct LGState{Lx,Ly} <: AbstractState
    Γ::Vector{AbstractTensorMap}
    λ::Vector{AbstractTensorMap}
    nnsites::Tuple
    nntable::Dict{Tuple,Tuple}
    λindex::Tuple
    pspace::ElementarySpace
    function LGState(Latt::AbstractLattice)
        return new{size(Latt)...}(Vector{AbstractTensorMap}(),Vector{AbstractTensorMap}(),(),Dict{Tuple,Tuple}(),(),ℂ^1)
    end
end

trivial(::GradedSpace{I, D}) where {I, D} = GradedSpace{I,D}(TensorKit.SortedVectorDict(one(I) => 1), false)
trivial(::ComplexSpace) = ℂ^1

function _direction(R::AbstractVector)
    @assert norm(R) ≠ 0
    if dot(R,[0,1]) == 0
        if dot(R,[1,0]) > 0
            return RIGHT()
        else
            return LEFT()
        end
    end 
    if dot(R,[1,0]) == 0
        if dot(R,[0,1]) > 0
            return UP()
        else
            return DOWN()
        end
    end
end
function build_direction_table(Latt::AbstractLattice)
    nbs = neighborsites_pbc(Latt)
    DM = Dict{Tuple,Tuple}()
    for n in [(i,[0,0]) for i in 1:length(Latt)],nn in nbs[n[1]]
        DM[(n,_direction(coordinate(Latt,nn...) - coordinate(Latt,n...)))] = nn
    end
    return DM
end

function addIntr2!(H::Hamiltonian,nb::Tuple,J::Matrix)
    !haskey(H.H2,nb) && (H.H2[nb] = zeros(3,3))
    H.H2[nb] += J
    return H
end

function addIntr1!(H::Hamiltonian,i::Int64,h::Vector)
    !haskey(H.H1,i) && (H.H1[i] = zeros(3))
    H.H1[i] += h
    return H
end

function addIntr1!(H::Hamiltonian,sites::Union{UnitRange,Vector},h::Vector)
    for i in sites
        addIntr1!(H,i,h)
    end
    return H
end

function addIntr2!(H::Hamiltonian,nbs::Vector,J::Matrix)
    for nb in nbs
        addIntr2!(H,nb,J)
    end
    return H
end

function initialize!(Latt::AbstractLattice, H::Hamiltonian)
    isnothing(H.nnnpath) && (H.nnnpath = Dict{Tuple,Tuple}())

    nbs = ineighbor(Latt)
    # nnnnb = filter(x -> x ∉ nbs, collect(keys(H.H2)))
    # osites = [(i,[0,0]) for i in 1:length(Latt)]
    # its = map(x -> map(z -> z[2], filter(y -> y[1] == x,nnnnb)),osites)
    # paths = map(x -> findpath(Latt,osites[x],its[x]), 1:length(Latt))

    # for i in 1:length(Latt),j in eachindex(its[i])
    #     H.nnnpath[((i,[0,0]),its[j])] = Tuple(paths[i][j])
    # end
    for nb in keys(H.H2)
        nb ∈ nbs && continue
        H.nnnpath[nb] = Tuple(findpath(Latt,nb...))
    end
    return H
end

function initialize!(Latt::AbstractLattice,ψ::LGState,pspace::ElementarySpace,aspace::ElementarySpace = trivial(pspace))
    nbs = ineighbor(Latt)
    ψ.Γ = [rand(aspace ⊗ aspace ⊗ pspace, aspace ⊗ aspace) for _ in 1:length(Latt)]
    ψ.λ = [isometry(aspace, aspace) for _ in 1:length(nbs)]
    ψ.nnsites = Tuple(neighborsites_pbc(Latt))
    ψ.pspace = pspace
    ψ.nntable = build_direction_table(Latt)

    nn2λ = Dict()
    for (i,n) in enumerate(nbs)
        nn2λ[n] = i
        nn2λ[_nn_reverse(n)] = i
    end
    
    λindex = []
    for i in 1:length(Latt)
        push!(λindex,_λindex(ψ,i,nn2λ))
    end
    ψ.λindex = Tuple(λindex)

    return ψ
end

function _nn_reverse(a::Tuple)
    return ((a[2][1],[0,0]),(a[1][1],-a[2][2]))
end

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

Latt = PeriSqua(8,8)

H = Hamiltonian()
addIntr2!(H,ineighbor(Latt),diagm(ones(3)))
addIntr1!(H,1:length(Latt),ones(3))
addIntr2!(H,ineighbor(Latt;level = 2),diagm(ones(3)))


ψ = LGState(Latt)

@time begin
initialize!(Latt,H)
initialize!(Latt,ψ,ℂ^2)
end

replace!(ψ,ψ[1][1],1)
replace!(ψ,ψ[1][2],1)
replace!(ψ,ψ[1][2][1],1,RIGHT())
ψ.λindex[36][1] == ψ.λindex[37][4]