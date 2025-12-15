
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

