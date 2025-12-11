using TensorKit
include("../src/iPEPS.jl")

abstract type AbstractDirection end

struct Right <: AbstractDirection end
struct Up <: AbstractDirection end
struct Left <: AbstractDirection end
struct Down <: AbstractDirection end

function diagm(A::Pair{Int64, Vector{T}}) where T
    L = abs(A.first) + length(A.second)
    B = zeros(T,L,L)
    for i in 1:length(A.second)
        B[(A.first > 0 ? (i,abs(A.first) + i) : (abs(A.first) + i,i))...] = A.second[i]
    end
    return B
end
diagm(A::Vector) = diagm(0 => A)
O1_2_O2_l(A::AbstractTensorMap,pspace::ElementarySpace) = TensorMap(kron(convert(Array,A),diagm(ones(dim(pspace)))), pspace ⊗ pspace,pspace ⊗ pspace)
O1_2_O2_r(A::AbstractTensorMap,pspace::ElementarySpace) = TensorMap(kron(diagm(ones(dim(pspace))),convert(Array,A)), pspace ⊗ pspace,pspace ⊗ pspace)

function normalize!(Latt::AbstractLattice, ψ::Dict)
    for i in 1:length(Latt)
        ψ["λu"][i] /= norm(ψ["λu"][i])
        ψ["λr"][i] /= norm(ψ["λr"][i])
        ψ["Γ"][i] /= norm(ψ["Γ"][i])
    end
end

function λs(ψ::Dict,Latt::AbstractLattice,i::Int64)
    Lx,Ly = size(Latt)
    ind = Latt[i][2] + [1,1]
    indu = ind[1], mod(ind[2] - 1 + Ly - 1, Ly) + 1
    indl = mod(ind[1] - 1 + Lx - 1, Lx) + 1, ind[2]
    ψ["λr"][ind...], ψ["λu"][ind...], ψ["λu"][indu...], ψ["λr"][indl...]
end

function pad(S::AbstractTensorMap, new_space::VectorSpace)
    # 1. 创建一个全零的新张量，定义在更大的空间上
    # S 通常是 (Space) -> (Space)
    S_padded = zeros(eltype(S), new_space, new_space)
    
    # 2. 遍历旧张量的 block，把数据拷过去
    # 注意：只有当 new_space 包含 old_space 的所有 sector 时才有效
    for (c, b) in blocks(S)
        # 获取新张量对应 sector 的 block 引用
        # 注意：这里假设 S_padded 也有 sector c
        # if hasblock(S_padded, c)
        b_padded = block(S_padded, c)
        
        # 拷贝数据 (假设维度是从左上角对齐)
        dims = size(b)
        b_padded[1:dims[1], 1:dims[2]] = b
        # end
    end
    
    return S_padded
end

function diff(A::AbstractTensorMap, B::AbstractTensorMap)
    if rank(A) == rank(B)
        return norm(A - B)
    elseif rank(A) > rank(B)
        return norm(A - pad(B,codomain(A)))
    else
        return norm(B - pad(A,codomain(B)))
    end
end

    # (mod.(Latt[i][2] - [1,0] .+ size(Latt)[1] .- 1, size(Latt)[1]) + [1,1])...]

# λs(ψ::Dict,Latt::AbstractLattice,i::Int64) = ψ["λr"][i], ψ["λu"][i], ψ["λu"][mod(i - 1 + size(Latt)[2] - 1,size(Latt)[2]) + 1], ψ["λr"][mod(i - 1 + size(Latt)[1] - 1,size(Latt)[1]) + 1]
# map(f -> f(ψ,Latt,i), (λr,λu,λl,λd))
# λsid(Latt::AbstractLattice,i::Int64) = 
# λrid(::AbstractLattice,i::Int64) = ψ["λr"][i]
# λuid(::AbstractLattice,i::Int64) = ψ["λu"][i]
# λlid(Latt::AbstractLattice,i::Int64) = ψ["λr"][mod(i + size(Latt)[1] - 1,size(Latt)[1]) + 1]
# λdid(Latt::AbstractLattice,i::Int64) = ψ["λu"][mod(i + size(Latt)[2] - 1,size(Latt)[2]) + 1]


function Γcontractlr(Γl′::AbstractTensorMap, Γr′::AbstractTensorMap)
    @tensor tmp[-1,-2,-3,-4,-5;-6,-7,-8] ≔ Γl′[1,-3,-5,-7,-8] * Γr′[-1,-2,-4,-6,1]
    return tmp
end
# function action(tmp::AbstractTensorMap, O::AbstractTensorMap)
#     @tensor tmp′[-1,-2,-3,-4,-5;-6,-7,-8] ≔ tmp[-1,-2,-3,1,2,-6,-7,-8] * O[-4,-5,1,2]
#     return tmp′
# end

function actionlr(tmp::AbstractTensorMap, O::AbstractTensorMap)
    @tensor tmp′[-1,-2,-3,-4,-5;-6,-7,-8] ≔ tmp[-1,-2,-3,1,2,-6,-7,-8] * O[-5,-4,2,1]
    return tmp′
end

function actionud(tmp::AbstractTensorMap, O::AbstractTensorMap)
    @tensor tmp′[-1,-2,-3,-4,-5;-6,-7,-8] ≔ tmp[-1,-2,-3,1,2,-6,-7,-8] * O[-4,-5,1,2]
    return tmp′
end

function Γcontractud(Γd′::AbstractTensorMap,Γu′::AbstractTensorMap)
    @tensor tmp[-1,-2,-3,-4,-5;-6,-7,-8] ≔ Γu′[-2,-3,-5,1,-8] * Γd′[-1,1,-4,-6,-7]
    return tmp
end




function invl(Γl′::AbstractTensorMap, λlu::AbstractTensorMap, λll::AbstractTensorMap, λld::AbstractTensorMap)
    iλlu, iλll, iλld = map(x -> inv(x),(λlu, λll, λld))
    @tensor Γl′[-1,-2,-3;-4,-5] = iλlu[-2,2] * iλll[5,-5] * iλld[4,-4] * Γl′[-1,2,-3,4,5]
    return Γl′
end

function invr(Γr′::AbstractTensorMap, λrr::AbstractTensorMap, λru::AbstractTensorMap, λrd::AbstractTensorMap)
    iλrr, iλru, iλrd = map(x -> inv(x),(λrr, λru, λrd))
    @tensor Γr′[-1,-2,-3;-4,-5] = iλrr[-1,1] * iλru[-2,2] * iλrd[4,-4] * Γr′[1,2,-3,4,-5]
    return Γr′
end

function invu(Γu′::AbstractTensorMap, λur::AbstractTensorMap, λuu::AbstractTensorMap, λul::AbstractTensorMap)
    iλur, iλuu, iλul = map(x -> inv(x),(λur, λuu, λul))
    @tensor Γu′[-1,-2,-3;-4,-5] = iλur[-1,1] * iλuu[-2,2] * iλul[5,-5] * Γu′[1,2,-3,-4,5]
    return Γu′
end
function invd(Γd′::AbstractTensorMap, λdr::AbstractTensorMap, λdd::AbstractTensorMap, λdl::AbstractTensorMap)
    iλdr, iλdd, iλdl = map(x -> inv(x),(λdr, λdd, λdl))
    @tensor Γd′[-1,-2,-3;-4,-5] = iλdr[-1,1] * iλdd[4,-4] * iλdl[5,-5] * Γd′[1,-2,-3,4,5]
    return Γd′
end

function _SUupdate!(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::Right, SUalgo::Dict)
    λlr, λlu, λld, λll= λs(ψ,Latt,i)
    λrr, λru, λrd, λrl= λs(ψ,Latt,j)
    Γl = ψ["Γ"][(Latt[i][2] + [1,1])...]
    Γr = ψ["Γ"][(Latt[j][2] + [1,1])...]
    @assert λlr == λrl

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    tmp′ = actionlr(Γcontractlr(Γl′,Γr′),O) |> x -> x + SUalgo["noise"] * rand(space(x))

    Γr′,Λ,Γl′,ϵ_trunc = tsvd(tmp′,(1,2,4,6),(3,5,7,8);trunc = truncbelow(SUalgo["ϵ"]) & truncdim(SUalgo["D"]))

    Γl′ = permute(Γl′,(1,2,3),(4,5))
    Γr′ = permute(Γr′,(1,2,3),(4,5))
    Λ = normalize(Λ)

    ϵ_λ = diff(ψ["λr"][(Latt[i][2] + [1,1])...], Λ)
    ψ["λr"][(Latt[i][2] + [1,1])...] = Λ


    ψ["Γ"][(Latt[i][2] + [1,1])...] = invl(Γl′,λlu, λll, λld)
    ψ["Γ"][(Latt[j][2] + [1,1])...] = invr(Γr′,λrr, λru, λrd)
    return ψ, ϵ_trunc, ϵ_λ
end

function _SUupdate!(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::Up, SUalgo::Dict)
    λur, λuu, λud, λul= λs(ψ,Latt,j)
    λdr, λdu, λdd, λdl= λs(ψ,Latt,i)
    Γu = ψ["Γ"][(Latt[j][2] + [1,1])...]
    Γd = ψ["Γ"][(Latt[i][2] + [1,1])...]
    @assert λud == λdu

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    tmp′ = actionud(Γcontractud(Γd′,Γu′), O) |> x -> x + SUalgo["noise"] * rand(space(x))

    Γu′,Λ,Γd′,ϵ_trunc = tsvd(tmp′,(2,3,5,8),(1,4,6,7);trunc = truncbelow(SUalgo["ϵ"]) & truncdim(SUalgo["D"]))
    Γu′ = permute(Γu′,(1,2,3),(5,4))
    Γd′ = permute(Γd′,(2,1,3),(4,5))
    Λ = normalize(Λ)
    ϵ_λ = diff(ψ["λu"][(Latt[i][2] + [1,1])...],Λ)

    ψ["λu"][(Latt[i][2] + [1,1])...] = Λ
    ψ["Γ"][(Latt[j][2] + [1,1])...] = invu(Γu′,λur, λuu, λul)
    ψ["Γ"][(Latt[i][2] + [1,1])...] = invd(Γd′,λdr, λdd, λdl)
    return ψ, ϵ_trunc, ϵ_λ
end

function λΓcontract(Γ::AbstractTensorMap, 
    λlr::AbstractTensorMap, λlu::AbstractTensorMap, λld::AbstractTensorMap, λll::AbstractTensorMap)
    @tensor Γl′[-1,-2,-3;-4,-5] ≔ λlr[-1,1] * λlu[-2,2] * λll[5,-5] * λld[4,-4] * Γ[1,2,-3,4,5]
    return Γl′
end

function _SUupdate!(ψ::Dict,H::Dict,Latt::AbstractLattice, SUalgo::Dict;pspace = ℂ^2)
    ϵ_trunc_tol = 0.0
    ϵ_λ_tol = 0.0
    for (ind,((i,j),v)) in enumerate(H["sites2"])
        R = Latt[j][2] - Latt[i][2] + v
        
        Heff = H["H2"]

        if i in H["sites1"]
            H1 = O1_2_O2_l(H["H1"],pspace) / H["sites1nb"][i]
            Heff += H1
        end

        if j in H["sites1"]
            H1 = O1_2_O2_r(H["H1"],pspace) / H["sites1nb"][j]
            Heff += H1
        end

        O = exp(- SUalgo["τ"] * Heff)
        if R == [1,0]
            _,ϵ_trunc,ϵ_λ = _SUupdate!(ψ,O,Latt,i,j,Right(),SUalgo)
        elseif R == [0,1]
            _,ϵ_trunc,ϵ_λ = _SUupdate!(ψ,O,Latt,i,j,Up(),SUalgo)
        end
        ϵ_trunc_tol += ϵ_trunc
        ϵ_λ_tol += ϵ_λ
    end
    return ϵ_trunc_tol / 2length(ψ["Γ"]), ϵ_λ_tol / 2length(ψ["Γ"])
end

function SUupdate!(ψ::Dict,H::Dict,Latt::AbstractLattice, SUalgo::Dict)
    for τ in SUalgo["τs"]
        SUalgo["τ"] = τ
        for i in 1:SUalgo["N"]
            SUalgo["noise"] = 0.0
            ϵ,tol = _SUupdate!(ψ,H,Latt,SUalgo)
            tol < τ * SUalgo["tol"] && break
            if i == SUalgo["N"]
                println("SU update not converged!")
            end
        end
    end
end

function _calObs2(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::Right)
    λlr, λlu, λld, λll= λs(ψ,Latt,i)
    λrr, λru, λrd, λrl= λs(ψ,Latt,j)
    Γl = ψ["Γ"][(Latt[i][2] + [1,1])...]
    Γr = ψ["Γ"][(Latt[j][2] + [1,1])...]
    @assert λlr == λrl

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    tmp = Γcontractlr(Γl′,Γr′)
    tmp′ = actionlr(tmp,O)

    return real(_inner(tmp′,tmp))
end

function _calObs2(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::Up)
    λur, λuu, λud, λul= λs(ψ,Latt,j)
    λdr, λdu, λdd, λdl= λs(ψ,Latt,i)
    Γu = ψ["Γ"][(Latt[j][2] + [1,1])...]
    Γd = ψ["Γ"][(Latt[i][2] + [1,1])...]
    @assert λud ≈ λdu

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    tmp = Γcontractud(Γd′, Γu′)
    tmp′ = actionud(tmp, O)

    return real(_inner(tmp′,tmp))
end

function stepτ(A::Float64)
    if A > 1e-3
        return 1e-1
    elseif 1e-3 ≥ A > 1e-5
        return 1e-2
    elseif 1e-5 ≥ A > 1e-7
        return 1e-3
    else
        return 1e-4
    end
end

# function changeτ!(H::Dict,τ::Float64)
#     H["expH"] = [exp(-τ*H["H"]) for _ in eachindex(H["sites"])]
# end

_inner(tmp′::AbstractTensorMap,tmp::AbstractTensorMap) = @tensor tmp′[1,2,3,4,5,6,7,8] * tmp'[6,7,8,1,2,3,4,5]

function _calObs2(ψ::Dict,O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, v::Vector)
    R = Latt[j][2] - Latt[i][2] + v
    if R == [1,0]
        return _calObs2(ψ,O,Latt,i,j,Right())
    elseif R == [0,1]
        return _calObs2(ψ,O,Latt,i,j,Up())
    end
end

function measure2(ψ::Dict,H::Dict)
    E = 0.0
    for ((i,j),v) in H["sites2"]
        E += _calObs2(ψ,H["H2"],Latt,i,j,v)
    end
    return E
end

Lx = 2
Ly = 2
Latt = PeriSqua(Lx,Ly)
@save "Heisenberg/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2

pspace = ℂ^2
aspace = ℂ^1

ψ = Dict(
    "Γ" => [rand(ComplexF64, aspace ⊗ aspace ⊗ pspace, aspace ⊗ aspace) for i in 1:Lx, j in 1:Ly],
    "λu" => [normalize(rand(ComplexF64, aspace, aspace)) for i in 1:Lx, j in 1:Ly],
    "λr" => [normalize(rand(ComplexF64, aspace, aspace)) for i in 1:Lx, j in 1:Ly]
)


SS = let 
    Sx = [0 1;1 0] / 2
    Sy = [0 -1im;1im 0] / 2
    Sz = [1 0;0 -1] / 2
    SxSx = TensorMap(kron(Sx,Sx),pspace ⊗ pspace, pspace ⊗ pspace)
    SySy = TensorMap(kron(Sy,Sy),pspace ⊗ pspace, pspace ⊗ pspace)
    SzSz = TensorMap(kron(Sz,Sz),pspace ⊗ pspace, pspace ⊗ pspace)
    SxSx + SySy + SzSz
end

Sv = let 
    Sx = [0 1;1 0] / 2
    Sy = [0 -1im;1im 0] / 2
    Sz = [1 0;0 -1] / 2
    TensorMap(cat(map(x -> reshape(x,2,1,2),(Sx,Sy,Sz))...;dims = 2), pspace ⊗ ℂ^3, pspace)
end

h = [0.0,1.0,0.0]
Sh = let 
    Sx = [0 1;1 0] / 2
    Sy = [0 -1im;1im 0] / 2
    Sz = [1 0;0 -1] / 2
    TensorMap(- (h[1]*Sx + h[2]*Sy + h[3]*Sz), pspace, pspace)
end

params = (J = 0.0, h = 10.0)
H = Dict(
    "sites1" => [1,],
    "sites2" => neighbor_pbc(Latt;issort = false),
    "H2" => params.J * SS,
    "H1" => params.h *Sh
)

H["sites1nb"] = map(x -> length(filter(y -> x in y[1],H["sites2"])), H["sites1"])

SUalgo = Dict(
    "ϵ" => 1e-8,
    "D" => D,
    "tol" => 1e-3,
    "N" => 3000,
    "τs" => [0.1,0.01,0.001,0.0001],
    "τ" => 0.0
)

SUupdate!(ψ,H,Latt,SUalgo)
@show measure2(ψ,H) / length(Latt)
Obs = Dict(
    "S" => []
)
for i in 1:length(Latt)
    O = Sv
    λr, λu, λd, λl= λs(ψ,Latt,i)
    Γ = ψ["Γ"][(Latt[i][2] + [1,1])...]
    Γ′ = λΓcontract(Γ, λr, λu, λd, λl)
    @tensor tmp[-1] ≔ Γ′[1,2,3,5,6] * O[4,-1,3] * Γ′'[5,6,1,2,4]
    push!(Obs["S"], real(convert(Array,tmp)))
end

@save "Heisenberg/data/Obs_$(Lx)x$(Ly)_$(D)_$(params).jld2" Obs
