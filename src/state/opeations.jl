
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
    return DΓ,C,UΓ
end

dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::RIGHT) where {T,S} = @tensor C[-1,-2,-3;-4,-5] ≔ A[1,-2,-4,-5] * B[-1,-3,1]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::LEFT) where {T,S} = @tensor C[-1,-2,-3;-4,-5] ≔ A[-1,-2,-4,1] * B[1,-3,-5]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::UP) where {T,S} = @tensor Γd′[-1,-2,-3;-4,-5] ≔ A[-1,1,-4,-5] * B[-2,-3,1]
dekernalize(A::AbstractTensorMap{T,S,2,2}, B::AbstractTensorMap, ::DOWN) where {T,S} = @tensor Γu′[-1,-2,-3;-4,-5] ≔ A[-1,-2,1,-5] * B[1,-3,-4]


function _swap!(ψ::LGState,i::Int64,j::Int64,::RIGHT,trunc::TensorKit.TruncationScheme)
    Γl, λlr, λlu, λld, λll = ψ[i]
    Γr, λrr, λru, λrd, λrl = ψ[j]

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    LΓ,K,RΓ = kernalize(Γl′,Γr′,RIGHT())
    SW = _swap_gate(ψ.pspace)
    K = action(K,SW)
    U,Λ,V,ϵ = tsvd(K,(1,2),(3,4);trunc = trunc)
    Γl′ = dekernalize(LΓ,V,RIGHT())
    Γr′ = dekernalize(RΓ,U,LEFT())
    Λ = normalize(Λ)
    replace!(ψ,invl(Γl′,λlu, λll, λld),i)
    replace!(ψ,invr(Γr′,λrr, λru, λrd),j)
    replace!(ψ,Λ,i,RIGHT())
    return ψ
end

function _swap!(ψ::LGState,i::Int64,j::Int64,::UP,trunc::TensorKit.TruncationScheme)
    Γu, λur, λuu, λud, λul = ψ[j]
    Γd, λdr, λdu, λdd, λdl = ψ[i]

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    DΓ,K,UΓ = kernalize(Γd′,Γu′,UP())
    SW = _swap_gate(ψ.pspace)
    K = action(K,SW)
    U,Λ,V,_ = tsvd(K,(1,2),(3,4);trunc = trunc)
    Γd′ = dekernalize(DΓ,V,UP())
    Γu′ = dekernalize(UΓ,U,DOWN())
    Λ = normalize(Λ)

    replace!(ψ,Λ,i,UP())
    replace!(ψ,invu(Γu′,λur, λuu, λul),j)
    replace!(ψ,invd(Γd′,λdr, λdd, λdl),i)
    return ψ
end

_swap!(ψ::LGState,i::Int64,j::Int64,::DOWN,trunc::TensorKit.TruncationScheme) = _swap!(ψ,j,i,UP(),trunc)
_swap!(ψ::LGState,i::Int64,j::Int64,::LEFT,trunc::TensorKit.TruncationScheme) = _swap!(ψ,j,i,RIGHT(),trunc)


function _swap!(ψ::LGState,path::Tuple,trunc::TensorKit.TruncationScheme)
    for k in 1:length(path)-1
        (i,vi),(j,vj) = path[k:k+1]
        _swap!(ψ,i,j,ψ.nn2d[((i,[0,0]),(j,vj-vi))],trunc)
    end
    return ψ
end

