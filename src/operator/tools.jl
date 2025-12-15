O1_2_O2_l(A::AbstractTensorMap,pspace::ElementarySpace) = kron(A,isometry(pspace,pspace))
O1_2_O2_r(A::AbstractTensorMap,pspace::ElementarySpace) = kron(isometry(pspace,pspace),A)

_swap_gate(pspace::ElementarySpace) = permute(id(pspace ⊗ pspace) , (2, 1), (3, 4))

# action(K::AbstractTensorMap{T₁,S₁,3,1},O::AbstractTensorMap{T₂,S₂,2,2},::RIGHT) where {T₁,S₁,T₂,S₂} = @tensor C[-1,-2,-3;-4] ≔ K[-1,1,2,-4] * O[-3,-2,2,1]
action(K::AbstractTensorMap{T₁,S₁,3,1},O::AbstractTensorMap{T₂,S₂,2,2}) where {T₁,S₁,T₂,S₂} = @tensor C[-1,-2,-3;-4] ≔ K[-1,1,2,-4] * O[-3,-2,2,1]
_inner(tmp′::AbstractTensorMap{T,S,3,1},tmp::AbstractTensorMap{T,S,3,1}) where {T,S} = @tensor tmp′[1,3,4,2] * tmp'[2,1,3,4]
function _inner(Γ₁::AbstractTensorMap, O::AbstractTensorMap{T,S,1,1}, Γ₂::AbstractTensorMap) where {T,S}
    return @tensor Γ₁[1,2,3,5,6] * O[4,3] * Γ₂'[5,6,1,2,4]
end