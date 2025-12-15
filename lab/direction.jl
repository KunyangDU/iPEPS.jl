using TensorKit,Random
include("../src/iPEPS.jl")

abstract type AbstractAlgorithm end
mutable struct SimpleUpdate <: AbstractAlgorithm
    trunc::TensorKit.TruncationScheme
    tol::Float64
    N::Int64
    τs::Vector{Float64}
    τ::Float64
    noise::Float64
end

abstract type AbstractOperator end
mutable struct Hamiltonian <: AbstractOperator
    H2::Dict{Tuple,TensorMap}
    H1::Dict{Int64,TensorMap}
    nnnpath::Union{Nothing,Dict{Tuple,Tuple}}
    coordination::Tuple
    pspace::ElementarySpace
    function Hamiltonian(H2::Dict{Tuple,TensorMap},
    H1::Dict{Int64,TensorMap})
        return new(H2,H1,nothing,(),ℂ^1)
    end
    Hamiltonian() = new(Dict{Tuple,TensorMap}(),Dict{Int64,TensorMap}(),nothing,(),ℂ^1)
end

mutable struct Observable <: AbstractOperator
    O2::Union{Dict{Tuple,Vector},Dict{Tuple,TensorMap}}
    O1::Union{Dict{Int64,Vector},Dict{Int64,TensorMap}}
    nnnpath::Union{Nothing,Dict{Tuple,Tuple}}
    values::Dict 
    Observable() = new(Dict{Tuple,Vector}(), Dict{Int64,Vector}(),nothing,Dict())
end

abstract type AbstractState end
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
    nd2n = Dict{Tuple,Tuple}()
    nn2d = Dict{Tuple,AbstractDirection}()
    for n in [(i,[0,0]) for i in 1:length(Latt)],nn in nbs[n[1]]
        d = _direction(coordinate(Latt,nn...) - coordinate(Latt,n...))
        nd2n[(n,d)] = nn
        nn2d[(n,nn)] = d
        nn2d[(nn,n)] = d'
    end
    return nd2n,nn2d
end

function addIntr2!(H::Hamiltonian,nb::Tuple,J::TensorMap)
    if haskey(H.H2,nb)
        H.H2[nb] += J
    else
        H.H2[nb] = J
    end
    return H
end

function addIntr1!(H::Hamiltonian,i::Int64,h::TensorMap)
    if haskey(H.H1,i)
        H.H1[i] += h
    else
        H.H1[i] = h
    end
    return H
end

function addIntr2!(H::Hamiltonian,nbs::Vector,J::TensorMap)
    for nb in nbs
        addIntr2!(H,nb,J)
    end
    return H
end

function addIntr1!(H::Hamiltonian,sites::Union{UnitRange,Vector},h::TensorMap)
    for i in sites
        addIntr1!(H,i,h)
    end
    return H
end

function initialize!(Latt::AbstractLattice, H::Hamiltonian, pspace::ElementarySpace)
    nbs = ineighbor(Latt)
    # nnnnb = filter(x -> x ∉ nbs, collect(keys(H.H2)))
    # osites = [(i,[0,0]) for i in 1:length(Latt)]
    # its = map(x -> map(z -> z[2], filter(y -> y[1] == x,nnnnb)),osites)
    # paths = map(x -> findpath(Latt,osites[x],its[x]), 1:length(Latt))

    # for i in 1:length(Latt),j in eachindex(its[i])
    #     H.nnnpath[((i,[0,0]),its[j])] = Tuple(paths[i][j])
    # end
    zs = zeros(Int64,length(Latt))
    for nb in keys(H.H2)
        zs[nb[1][1]] += 1
        zs[nb[2][1]] += 1
        nb ∈ nbs && continue
        isnothing(H.nnnpath) && (H.nnnpath = Dict{Tuple,Tuple}())
        H.nnnpath[nb] = Tuple(findpath(Latt,nb...))
    end

    H.coordination = Tuple(zs)

    H.pspace = pspace
    return H
end

function initialize!(Latt::AbstractLattice,ψ::LGState,pspace::ElementarySpace,aspace::ElementarySpace = trivial(pspace))
    nbs = ineighbor(Latt)
    ψ.Γ = [rand(ComplexF64, aspace ⊗ aspace ⊗ pspace, aspace ⊗ aspace) for _ in 1:length(Latt)]
    ψ.λ = [isometry(ComplexF64, aspace, aspace) for _ in 1:length(nbs)]
    ψ.nnsites = Tuple(neighborsites_pbc(Latt))
    ψ.pspace = pspace
    ψ.nntable,ψ.nn2d = build_direction_table(Latt)

    nn2λ = Dict()
    for (i,n) in enumerate(nbs)
        nn2λ[n] = i
        nn2λ[_nn_reverse(n)] = i
    end
    # ψ.nn2d = 
    
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


function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::UP, algo::SimpleUpdate)
    to = TimerOutput()
    Γu, (λur, λuu, λud, λul) = ψ[j]
    Γd, (λdr, λdu, λdd, λdl) = ψ[i]
    @assert λud == λdu

    @timeit to "λ-Γ contract" Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    @timeit to "λ-Γ contract" Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    @timeit to "kernalize" UΓ,K,DΓ = kernalize(Γd′,Γu′,UP())
    @timeit to "action" C = action(K,exp(- algo.τ * O),UP())
    @timeit to "measure" ΔE = real(_inner(action(K,O,UP()),K))
    @timeit to "svd" U,Λ,V,ϵ_trunc = tsvd(C,(1,2),(3,4);trunc = algo.trunc)
    @timeit to "dekernalize" Γd′ = dekernalize(DΓ,V,UP())
    @timeit to "dekernalize" Γu′ = dekernalize(UΓ,U,DOWN())
    # @timeit to "action" tmp′ = actionud(Γcontractud(Γd′,Γu′), O) |> x -> x + algo.noise * rand(space(x))

    # @timeit to "svd" Γu′,Λ,Γd′,ϵ_trunc = tsvd(tmp′,(2,3,5,8),(1,4,6,7);trunc = algo.trunc)
    # Γu′ = permute(Γu′,(1,2,3),(5,4))
    # Γd′ = permute(Γd′,(2,1,3),(4,5))
    Λ = normalize(Λ)
    ϵ_λ = diff(ψ[i][2][2],Λ)

    replace!(ψ,Λ,i,UP())
    replace!(ψ,invu(Γu′,λur, λuu, λul),j)
    replace!(ψ,invd(Γd′,λdr, λdd, λdl),i)

    return ψ, ϵ_trunc, ϵ_λ, ΔE, to
end

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::RIGHT, algo::SimpleUpdate)
    to = TimerOutput()
    Γl, (λlr, λlu, λld, λll) = ψ[i]
    Γr, (λrr, λru, λrd, λrl) = ψ[j]
    @assert λlr == λrl

    @timeit to "λ-Γ contract" Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    @timeit to "λ-Γ contract" Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    @timeit to "kernalize" LΓ,K,RΓ = kernalize(Γl′,Γr′,RIGHT())
    @timeit to "action" C = action(K,exp(- algo.τ * O),RIGHT())
    @timeit to "measure" ΔE = real(_inner(action(K,O,RIGHT()),K))
    @timeit to "svd" U,Λ,V,ϵ_trunc = tsvd(C,(1,2),(3,4);trunc = algo.trunc)
    @timeit to "dekernalize" Γl′ = dekernalize(LΓ,V,RIGHT())
    @timeit to "dekernalize" Γr′ = dekernalize(RΓ,U,LEFT())

    # @timeit to "action" tmp′ = actionlr(Γcontractlr(Γl′,Γr′),O) |> x -> x + algo.noise * rand(space(x))

    # @timeit to "svd" Γr′,Λ,Γl′,ϵ_trunc = tsvd(tmp′,(1,2,4,6),(3,5,7,8);trunc = algo.trunc)

    # Γl′ = permute(Γl′,(1,2,3),(4,5))
    # Γr′ = permute(Γr′,(1,2,3),(4,5))
    Λ = normalize(Λ)

    ϵ_λ = diff(ψ[i][2][1], Λ)
    
    replace!(ψ,Λ,i,RIGHT())
    replace!(ψ,invl(Γl′,λlu, λll, λld),i)
    replace!(ψ,invr(Γr′,λrr, λru, λrd),j)

    return ψ, ϵ_trunc, ϵ_λ, ΔE, to
end

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, algo::SimpleUpdate)
    to = TimerOutput()
    eO = exp(- algo.τ * O)
    Γ = ψ[i][1]
    @timeit to "action" @tensor Γ′[-1,-2,-3;-4,-5] ≔ Γ[-1,-2,1,-4,-5] * eO[-3,1]
    @timeit to "measure" ΔE = real(λΓcontract(ψ[i][1],ψ[i][2]...) |> x -> _inner(x,O,x))
    replace!(ψ,normalize(Γ′),i)
    return ψ, 0.0, 0.0, ΔE, to
end


function _SUupdate!(ψ::LGState, H::Hamiltonian, algo::SimpleUpdate)
    ϵ_trunc_tol = 0.0
    ϵ_λ_tol = 0.0
    E = 0.0
    sites1 = collect(keys(H.H1))
    to = TimerOutput()
    
    for (((i,vi),(j,vj)),Heff) in H.H2
        setdiff!(sites1,[i,j])
        @timeit to "build Heff" begin
            if i in keys(H.H1)
                H1 = O1_2_O2_l(H.H1[i],H.pspace) / H.coordination[i]
                Heff += H1
            end

            if j in keys(H.H1)
                H1 = O1_2_O2_r(H.H1[j],H.pspace) / H.coordination[j]
                Heff += H1
            end
        end

        norm(Heff) < 1e-12 && continue
        
        if haskey(ψ.nn2d,((i,vi),(j,vj)))
            @timeit to "update2!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,Heff,i,j,ψ.nn2d[(i,vi),(j,vj)], algo)
            merge!(to,localto;tree_point = ["update2!"])
        else
            # swap int
            ϵ_trunc = 0.0
            ϵ_λ = 0.0
        end

        ϵ_trunc_tol += ϵ_trunc
        ϵ_λ_tol += ϵ_λ
        E += ΔE
    end

    for i in sites1
        norm(H.H1[i]) < 1e-12 && continue
        @timeit to "update1!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,H.H1[i],i, algo)
        merge!(to,localto;tree_point = ["update1!"])
        ϵ_trunc_tol += ϵ_trunc
        ϵ_λ_tol += ϵ_λ
        E += ΔE
    end

    return ϵ_trunc_tol / length(ψ), ϵ_λ_tol / length(ψ), E, to
end

function SU!(ψ::LGState, H::Hamiltonian, algo::SimpleUpdate;
    showperstep::Int64 = 500)
    to = TimerOutput()
    for τ in algo.τs
        tmpto = TimerOutput()
        algo.τ = τ
        E = 0.0
        ΔE = 0.0
        tol = 0.0
        ϵ = 0.0
        for i in 1:algo.N
            ϵ,tol,E′,localto = _SUupdate!(ψ,H,algo)
            ΔE,E = E′ - E, E′
            merge!(tmpto,localto)
            tol < τ * algo.tol && break
            if i == algo.N
                println("SimpleUpdate update not converged!")
            end
            if mod(i,showperstep) == 0
                show(tmpto;title = "$(i)/$(algo.N)\nτ = $(τ)")
                println("\nE = $(E), ΔE/|E| = $(ΔE / abs(E)), TruncErr = $(ϵ), λErr = $(tol)")
                # print("\n")
            end
        end
        merge!(to,tmpto)
        show(to;title = "Simple Update\n -> $(τ)")
        println("\nE = $(E), ΔE/|E| = $(ΔE / abs(E)), TruncErr = $(ϵ), λErr = $(tol)")
    end
end

function initialize!(Latt::AbstractLattice, O::Observable)

    nbs = ineighbor(Latt)

    for nb in keys(O.O2)
        nb ∈ nbs && continue
        isnothing(O.nnnpath) && (O.nnnpath = Dict{Tuple,Tuple}())
        O.nnnpath[nb] = Tuple(findpath(Latt,nb...))
    end

    return O
end

function addObs2!(O::Observable,nb::Tuple,J::TensorMap)
    !haskey(O.O2,nb) && (O.O2[nb] = TensorMap[])
    push!(O.O2[nb], J)
    return O
end

function addObs1!(O::Observable,i::Int64,h::TensorMap)
    !haskey(O.O1,i) && (O.O1[i] = TensorMap[])
    push!(O.O1[i], h)
    return O
end

function addObs2!(O::Observable,nbs::Vector,J::TensorMap)
    for nb in nbs
        addObs2!(O,nb,J)
    end
    return O
end

function addObs1!(O::Observable,sites::Union{UnitRange,Vector},h::TensorMap)
    for i in sites
        addObs1!(O,i,h)
    end
    return O
end

# function densify(A::Vector,codom::ProductSpace = codomain(A[1]), dom::ProductSpace = codomain(A[1]))
#     N = length(A)
#     return TensorMap(cat(map(x -> reshape(convert(Array,x),dim.(codom)...,1,dim.(dom)...),A)...;dims = length(codom) + 1), codom ⊗ ℂ^N, dom)
#     # return TensorMap(cat(map(x -> reshape(x,dim(codom),1,dim(dom)),convert.(Array,A))...;dims = 2), codom ⊗ ℂ^N, dom)
# end

# function densify!(O::Observable)
#     vsd = (O.O1)
#     O.O1 = Dict{Int64,TensorMap}()
#     for (i,os) in vsd
#         O.O1[i] = densify(os)
#     end

#     vsd = O.O2
#     O.O2 = Dict{Tuple,TensorMap}()
#     for (nb,os) in vsd
#         O.O2[nb] = densify(os)
#     end
#     return O
# end

function calObs!(O::Observable, ψ::LGState)
    for (i,o) in O.O1
        O.values[i] = real(_calObs1(ψ,o,i))
    end
    for (((i,vi),(j,vj)),o) in O.O2
        O.values[((i,vi),(j,vj))] = real(_calObs2(ψ,o,i,j,ψ.nn2d[(i,vi),(j,vj)]))
    end
    return O
end



function measure(ψ::LGState,H::Hamiltonian)
    E = 0.0
    for ((sitei,sitej),J) in H.H2
        E += _calObs2(ψ,J,sitei,sitej)
    end
    for (i,h) in H.H1 
        E += _calObs1(ψ,h,i)
    end
    return E
end

function _calObs2(ψ::LGState, Os::Vector, i::Int64, j::Int64, ::RIGHT)
    Γl, (λlr, λlu, λld, λll) = ψ[i]
    Γr, (λrr, λru, λrd, λrl) = ψ[j]

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    _,K,_ = kernalize(Γl′,Γr′,RIGHT())
    return map(O -> real(_inner(action(K,O,RIGHT()),K)),Os)
end

function _calObs2(ψ::LGState, Os::Vector, i::Int64, j::Int64, ::UP)
    Γu, (λur, λuu, λud, λul) = ψ[j]
    Γd, (λdr, λdu, λdd, λdl) = ψ[i]

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    _,K,_ = kernalize(Γd′,Γu′,UP())
    return map(O -> real(_inner(action(K,O,UP()),K)),Os)
end

_calObs2(ψ::LGState,O::AbstractTensorMap, sitei::Tuple, sitej::Tuple) = _calObs2(ψ,[O,],sitei[1],sitej[1],ψ.nn2d[(sitei,sitej)])[1]

function _calObs1(ψ::LGState, os::Vector, i::Int64)
    Γ′ = λΓcontract(ψ[i][1],ψ[i][2]...)
    return map(o -> real(_inner(Γ′,o,Γ′)), os)
end


function kernalize(Γl′::AbstractTensorMap,Γr′::AbstractTensorMap,::RIGHT)
    L,LΓ = rightorth(Γl′,(1,3),(2,4,5))
    RΓ,R = leftorth(Γr′,(1,2,4),(3,5))
    R = permute(R,(1,2),(3,))
    @tensor C[-1,-2,-3;-4] ≔ L[1,-3,-4] * R[-1,-2,1]

    LΓ = permute(LΓ,(1,2),(3,4))
    RΓ = permute(RΓ,(1,2),(3,4))
    return LΓ,C,RΓ
end

function kernalize(Γd′::AbstractTensorMap,Γu′::AbstractTensorMap,::UP)
    D,DΓ = rightorth(Γd′,(2,3),(1,4,5))
    UΓ,U = leftorth(Γu′,(1,2,5),(3,4))
    # U = permute(U,(1,2),(3,))
    @tensor C[-1,-2,-3;-4] ≔ D[1,-3,-4] * U[-1,-2,1]

    UΓ = permute(UΓ,(1,2),(4,3))
    DΓ = permute(DΓ,(2,1),(3,4))
    return UΓ,C,DΓ
end

action(K::AbstractTensorMap{T₁,S₁,3,1},O::AbstractTensorMap{T₂,S₂,2,2},::RIGHT) where {T₁,S₁,T₂,S₂} = @tensor C[-1,-2,-3;-4] ≔ K[-1,1,2,-4] * O[-3,-2,2,1]
action(K::AbstractTensorMap{T₁,S₁,3,1},O::AbstractTensorMap{T₂,S₂,2,2},::UP) where {T₁,S₁,T₂,S₂} = @tensor C[-1,-2,-3;-4] ≔ K[-1,1,2,-4] * O[-2,-3,1,2]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::RIGHT) where {T,S} = @tensor C[-1,-2,-3;-4,-5] ≔ A[1,-2,-4,-5] * B[-1,-3,1]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::LEFT) where {T,S} = @tensor C[-1,-2,-3;-4,-5] ≔ A[-1,-2,-4,1] * B[1,-3,-5]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::UP) where {T,S} = @tensor Γd′[-1,-2,-3;-4,-5] ≔ A[-1,1,-4,-5] * B[-2,-3,1]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::DOWN) where {T,S} = @tensor Γu′[-1,-2,-3;-4,-5] ≔ A[-1,-2,1,-5] * B[1,-3,-4]

Latt = PeriSqua(2,2)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(1 * diagm(ones(3))))
# addIntr1!(H,1,LocalSpace.Sh(-[0,0,100]))
# addIntr1!(H,1,- 0.0001* LocalSpace.Sh([0,1,0]))
# addIntr2!(H,ineighbor(Latt;level = 2),LocalSpace.SJ(diagm(ones(3))))
end

ψ = LGState(Latt)

@time begin
initialize!(Latt,H,ℂ^2)
initialize!(Latt,ψ,ℂ^2)
end

D = 3

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    3000,
    [0.1,0.01,0.001],
    0.0,
    0.0
)

SU!(ψ,H,sualgo)

@show measure(ψ,H) / length(ψ)

O = let obs = Observable(), LocalSpace = TrivialSpinOneHalf
addObs1!(obs,1:length(Latt),LocalSpace.Sx)
addObs1!(obs,1:length(Latt),LocalSpace.Sy)
addObs1!(obs,1:length(Latt),LocalSpace.Sz)
addObs2!(obs,ineighbor(Latt),LocalSpace.SxSx)
addObs2!(obs,ineighbor(Latt),LocalSpace.SySy)
addObs2!(obs,ineighbor(Latt),LocalSpace.SzSz)
initialize!(Latt,obs)
end

calObs!(O,ψ)
O.values

