using TensorKit,Random
include("../src/iPEPS.jl")

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
    ψ.Γ = [rand(aspace ⊗ aspace ⊗ pspace, aspace ⊗ aspace) for _ in 1:length(Latt)]
    ψ.λ = [isometry(aspace, aspace) for _ in 1:length(nbs)]
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


function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::UP, SUalgo::Dict)
    # (i,vi),(j,vj) = sitei,sitej
    Γu, (λur, λuu, λud, λul) = ψ[j]
    Γd, (λdr, λdu, λdd, λdl) = ψ[i]
    @assert λud == λdu

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    tmp′ = actionud(Γcontractud(Γd′,Γu′), O) |> x -> x + SUalgo["noise"] * rand(space(x))

    Γu′,Λ,Γd′,ϵ_trunc = tsvd(tmp′,(2,3,5,8),(1,4,6,7);trunc = truncbelow(SUalgo["ϵ"]) & truncdim(SUalgo["D"]))
    Γu′ = permute(Γu′,(1,2,3),(5,4))
    Γd′ = permute(Γd′,(2,1,3),(4,5))
    Λ = normalize(Λ)
    ϵ_λ = diff(ψ[i][2][2],Λ)

    replace!(ψ,Λ,i,UP())
    replace!(ψ,invu(Γu′,λur, λuu, λul),j)
    replace!(ψ,invd(Γd′,λdr, λdd, λdl),i)

    return ψ, ϵ_trunc, ϵ_λ
end

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::RIGHT, SUalgo::Dict)
    # (i,vi),(j,vj) = sitei,sitej
    Γl, (λlr, λlu, λld, λll) = ψ[i]
    Γr, (λrr, λru, λrd, λrl) = ψ[j]
    @assert λlr == λrl

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    tmp′ = actionlr(Γcontractlr(Γl′,Γr′),O) |> x -> x + SUalgo["noise"] * rand(space(x))

    Γr′,Λ,Γl′,ϵ_trunc = tsvd(tmp′,(1,2,4,6),(3,5,7,8);trunc = truncbelow(SUalgo["ϵ"]) & truncdim(SUalgo["D"]))

    Γl′ = permute(Γl′,(1,2,3),(4,5))
    Γr′ = permute(Γr′,(1,2,3),(4,5))
    Λ = normalize(Λ)

    ϵ_λ = diff(ψ[i][2][1], Λ)
    replace!(ψ,Λ,i,RIGHT())
    replace!(ψ,invl(Γl′,λlu, λll, λld),i)
    replace!(ψ,invr(Γr′,λrr, λru, λrd),j)

    return ψ, ϵ_trunc, ϵ_λ
end


function _SUupdate!(ψ::LGState, H::Hamiltonian, SUalgo::Dict)
    ϵ_trunc_tol = 0.0
    ϵ_λ_tol = 0.0
    for (((i,vi),(j,vj)),Heff) in H.H2
        if i in keys(H.H1)
            H1 = O1_2_O2_l(H.H1[i],H.pspace) / H.coordination[i]
            Heff += H1
        end

        if j in keys(H.H1)
            H1 = O1_2_O2_r(H.H1[j],H.pspace) / H.coordination[j]
            Heff += H1
        end
        
        if haskey(ψ.nn2d,((i,vi),(j,vj)))
            _,ϵ_trunc,ϵ_λ = _SUupdate!(ψ,exp(- SUalgo["τ"] * Heff),i,j,ψ.nn2d[(i,vi),(j,vj)], SUalgo)
        else
            # swap int
            ϵ_trunc = 0.0
            ϵ_λ = 0.0
        end

        ϵ_trunc_tol += ϵ_trunc
        ϵ_λ_tol += ϵ_λ
    end

    return ϵ_trunc_tol / length(ψ), ϵ_λ_tol / length(ψ)
end

function SUupdate!(ψ::LGState, H::Hamiltonian, SUalgo::Dict)
    for τ in SUalgo["τs"]
        SUalgo["τ"] = τ
        for i in 1:SUalgo["N"]
            SUalgo["noise"] = 0.0
            ϵ,tol = _SUupdate!(ψ,H,SUalgo)
            tol < τ * SUalgo["tol"] && break
            if i == SUalgo["N"]
                println("SU update not converged!")
            end
        end
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

function densify(A::Vector,codom::ElementarySpace = space(A[1])[1], dom::ElementarySpace = space(A[1])[2]')
    N = length(A)
    return TensorMap(cat(map(x -> reshape(x,dim(codom),1,dim(dom)),convert.(Array,A))...;dims = 2), codom ⊗ ℂ^N, dom)
end

function densify!(O::Observable)
    vsd = (O.O1)
    O.O1 = Dict{Int64,TensorMap}()
    for (i,os) in vsd
        O.O1[i] = densify(os)
    end

    vsd = O.O2
    O.O2 = Dict{Tuple,TensorMap}()
    for (nb,os) in vsd
        O.O2[nb] = densify(os)
    end
    return O
end

function calObs!(O::Observable, ψ::LGState)
    for (i,o) in O.O1
        Γ′ = λΓcontract(ψ[i][1],ψ[i][2]...)
        tmp = _inner(Γ′,o,Γ′)
        O.values[i] = real(convert(Array,tmp))
    end
    return O
end

function measure2(ψ::LGState,H::Hamiltonian)
    E = 0.0
    for ((sitei,sitej),J) in H.H2
        E += _calObs2(ψ,J,sitei,sitej)
    end
    return E
end

function _calObs2(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::RIGHT)
    Γl, (λlr, λlu, λld, λll) = ψ[i]
    Γr, (λrr, λru, λrd, λrl) = ψ[j]

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    tmp = Γcontractlr(Γl′,Γr′)
    tmp′ = actionlr(tmp,O)

    return real(_inner(tmp′,tmp))
end

function _calObs2(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::UP)
    Γu, (λur, λuu, λud, λul) = ψ[j]
    Γd, (λdr, λdu, λdd, λdl) = ψ[i]

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    tmp = Γcontractud(Γd′, Γu′)
    tmp′ = actionud(tmp, O)

    return real(_inner(tmp′,tmp))
end


_calObs2(ψ::LGState,O::AbstractTensorMap, sitei::Tuple, sitej::Tuple) = _calObs2(ψ,O,sitei[1],sitej[1],ψ.nn2d[(sitei,sitej)])


Latt = PeriSqua(2,2)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
addIntr2!(H,ineighbor(Latt),LocalSpace.SJ(1 * diagm(ones(3))))
# addIntr1!(H,1,LocalSpace.Sh(- [0,0,100]))
addIntr1!(H,1,- 1* LocalSpace.Sh([0,0,1]))
# addIntr2!(H,ineighbor(Latt;level = 2),LocalSpace.SJ(diagm(ones(3))))
end

ψ = LGState(Latt)

@time begin
initialize!(Latt,H,ℂ^2)
initialize!(Latt,ψ,ℂ^2)
end

SUalgo = Dict(
    "ϵ" => 1e-8,
    "D" => 2,
    "tol" => 1e-4,
    "N" => 3000,
    "τs" => [0.01,0.001],
    # "τ" => 0.01,
    # "noise" => 0.0
)


SUupdate!(ψ,H,SUalgo)

@show measure2(ψ,H) / length(ψ)

O = let obs = Observable(), LocalSpace = TrivialSpinOneHalf
addObs1!(obs,1:length(Latt),LocalSpace.Sx)
addObs1!(obs,1:length(Latt),LocalSpace.Sy)
addObs1!(obs,1:length(Latt),LocalSpace.Sz)
initialize!(Latt,obs)
densify!(obs)
end

calObs!(O,ψ)
O.values

