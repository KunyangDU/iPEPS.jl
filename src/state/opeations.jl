
function kernalize(Γl′::AbstractTensorMap,Γr′::AbstractTensorMap,::RIGHT)
    L,LΓ = rightorth(Γl′,(1,3),(2,4,5))
    RΓ,R = leftorth(Γr′,(1,2,4),(3,5))
    R = permute(R,(1,2),(3,))
    return permute(LΓ,(1,2),(3,4)), (@tensor C[-1,-2,-3;-4] ≔ L[1,-3,-4] * R[-1,-2,1]) ,permute(RΓ,(1,2),(3,4))
end

function kernalize(Γd′::AbstractTensorMap,Γu′::AbstractTensorMap,::UP)
    D,DΓ = rightorth(Γd′,(2,3),(1,4,5))
    UΓ,U = leftorth(Γu′,(1,2,5),(3,4))
    return permute(DΓ,(2,1),(3,4)), (@tensor C[-1,-2,-3;-4] ≔ D[1,-3,-4] * U[-1,-2,1]) ,permute(UΓ,(1,2),(4,3))
end

dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::RIGHT) where {T,S} = @tensor C[-1,-2,-3;-4,-5] ≔ A[1,-2,-4,-5] * B[-1,-3,1]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::LEFT) where {T,S} = @tensor C[-1,-2,-3;-4,-5] ≔ A[-1,-2,-4,1] * B[1,-3,-5]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::UP) where {T,S} = @tensor Γd′[-1,-2,-3;-4,-5] ≔ A[-1,1,-4,-5] * B[-2,-3,1]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::DOWN) where {T,S} = @tensor Γu′[-1,-2,-3;-4,-5] ≔ A[-1,-2,1,-5] * B[1,-3,-4]

function _swap!(ψ::LGState,i::Int64,j::Int64,::RIGHT,algo::SimpleUpdate{FastSU})
    to = TimerOutput()
    Γl, λlr, _, _, _ = ψ[i]
    Γr, _, _, _, λrl = ψ[j]

    isλlr = inv(sqrt(λlr))
    isλrl = inv(sqrt(λrl))
    @timeit to "λ-Γ contract" (@tensor Γl′[-1,-2,-3;-4,-5] ≔ Γl[1,-2,-3,-4,-5] * isλlr[-1,1])
    @timeit to "λ-Γ contract" (@tensor Γr′[-1,-2,-3;-4,-5] ≔ Γr[-1,-2,-3,-4,1] * isλrl[1,-5])

    LΓ,K,RΓ = kernalize(Γl′,Γr′,RIGHT())
    SW = _swap_gate(ψ.pspace)
    K = action(K,SW)
    U,Λ,V,ϵ = tsvd(K,(1,2),(3,4);trunc = algo.trunc)
    Γl′ = dekernalize(LΓ,V,RIGHT())
    Γr′ = dekernalize(RΓ,U,LEFT())
    Λ = normalize(Λ)
    
    @timeit to "λ-Γ contract" (@tensor Γl′[-1,-2,-3;-4,-5] = Γl′[1,-2,-3,-4,-5] * Λ[-1,1])
    @timeit to "λ-Γ contract" (@tensor Γr′[-1,-2,-3;-4,-5] = Γr′[-1,-2,-3,-4,1] * Λ[1,-5])

    replace!(ψ,Λ,i,RIGHT())
    replace!(ψ,Γl′,i)
    replace!(ψ,Γr′,j)
    return ψ
end

function _swap!(ψ::LGState,i::Int64,j::Int64,::UP,algo::SimpleUpdate{FastSU})
    to = TimerOutput()
    Γu, _, _, λud, _ = ψ[j]
    Γd, _, λdu, _, _ = ψ[i]

    isλud = inv(sqrt(λud))
    isλdu = inv(sqrt(λdu))
    @timeit to "λ-Γ contract" (@tensor Γu′[-1,-2,-3;-4,-5] ≔ Γu[-1,-2,-3,1,-5] * isλud[1,-4])
    @timeit to "λ-Γ contract" (@tensor Γd′[-1,-2,-3;-4,-5] ≔ Γd[-1,1,-3,-4,-5] * isλdu[-2,1])

    DΓ,K,UΓ = kernalize(Γd′,Γu′,UP())
    SW = _swap_gate(ψ.pspace)
    K = action(K,SW)
    U,Λ,V,_ = tsvd(K,(1,2),(3,4);trunc = algo.trunc)
    Γd′ = dekernalize(DΓ,V,UP())
    Γu′ = dekernalize(UΓ,U,DOWN())
    Λ = normalize(Λ)

    @timeit to "λ-Γ contract" (@tensor Γu′[-1,-2,-3;-4,-5] = Γu′[-1,-2,-3,1,-5] * Λ[1,-4])
    @timeit to "λ-Γ contract" (@tensor Γd′[-1,-2,-3;-4,-5] = Γd′[-1,1,-3,-4,-5] * Λ[-2,1])

    replace!(ψ,Λ,i,UP())
    replace!(ψ,Γu′,j)
    replace!(ψ,Γd′,i)
    return ψ
end

function _swap!(ψ::LGState,i::Int64,j::Int64,::RIGHT,algo::SimpleUpdate{FullSU})
    Γl, λlr, λlu, λld, λll = ψ[i]
    Γr, λrr, λru, λrd, λrl = ψ[j]

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    LΓ,K,RΓ = kernalize(Γl′,Γr′,RIGHT())
    SW = _swap_gate(ψ.pspace)
    K = action(K,SW)
    U,Λ,V,ϵ = tsvd(K,(1,2),(3,4);trunc = algo.trunc)
    Γl′ = dekernalize(LΓ,V,RIGHT())
    Γr′ = dekernalize(RΓ,U,LEFT())
    Λ = normalize(Λ)
    replace!(ψ,invl(Γl′,λlu, λll, λld),i)
    replace!(ψ,invr(Γr′,λrr, λru, λrd),j)
    replace!(ψ,Λ,i,RIGHT())
    return ψ
end

function _swap!(ψ::LGState,i::Int64,j::Int64,::UP,algo::SimpleUpdate{FullSU})
    Γu, λur, λuu, λud, λul = ψ[j]
    Γd, λdr, λdu, λdd, λdl = ψ[i]

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    DΓ,K,UΓ = kernalize(Γd′,Γu′,UP())
    SW = _swap_gate(ψ.pspace)
    K = action(K,SW)
    U,Λ,V,_ = tsvd(K,(1,2),(3,4);trunc = algo.trunc)
    Γd′ = dekernalize(DΓ,V,UP())
    Γu′ = dekernalize(UΓ,U,DOWN())
    Λ = normalize(Λ)

    replace!(ψ,Λ,i,UP())
    replace!(ψ,invu(Γu′,λur, λuu, λul),j)
    replace!(ψ,invd(Γd′,λdr, λdd, λdl),i)
    return ψ
end

_swap!(ψ::LGState,i::Int64,j::Int64,::DOWN,algo::SimpleUpdate{T}) where T <: Union{FastSU,FullSU} = _swap!(ψ,j,i,UP(),algo)
_swap!(ψ::LGState,i::Int64,j::Int64,::LEFT,algo::SimpleUpdate{T}) where T <: Union{FastSU,FullSU} = _swap!(ψ,j,i,RIGHT(),algo)

# _swap!(ψ::LGState,i::Int64,j::Int64,d::AbstractDirection,trunc::TruncationScheme) = _swap!(ψ,i,j,d,SimpleUpdate(trunc))

function _swap!(ψ::LGState,path::Tuple,algo::SimpleUpdate)
    for k in 1:length(path)-1
        (i,vi),(j,vj) = path[k:k+1]
        _swap!(ψ,i,j,ψ.nn2d[((i,[0,0]),(j,vj-vi))],algo)
    end
    return ψ
end

_check_λ_index(ψ::LGState,i::Int64,j::Int64,::RIGHT) = (@assert ψ.λindex[i][1] == ψ.λindex[j][4] "λ index not compatible")
_check_λ_index(ψ::LGState,i::Int64,j::Int64,::LEFT) = (@assert ψ.λindex[i][4] == ψ.λindex[j][1] "λ index not compatible")
_check_λ_index(ψ::LGState,i::Int64,j::Int64,::UP) = (@assert ψ.λindex[i][2] == ψ.λindex[j][3] "λ index not compatible")
_check_λ_index(ψ::LGState,i::Int64,j::Int64,::DOWN) = (@assert ψ.λindex[i][3] == ψ.λindex[j][2] "λ index not compatible")
function _check_λ_index(ψ::LGState)
    for (((i,_),(j,_)),d) in ψ.nn2d
        _check_λ_index(ψ,i,j,d)
    end
end

function merge_λ!(ψ)
    for i in 1:length(ψ)
        replace!(ψ, λΓcontract(ψ[i]...),i)
    end
    return ψ
end

function dismerge_λ!(ψ)
    for i in 1:length(ψ)
        replace!(ψ, λΓcontract(ψ[i][1],inv.(ψ[i][2:5])...),i)
    end
    return ψ
end

