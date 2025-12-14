_inner(tmp′::AbstractTensorMap{T,S,5,3},tmp::AbstractTensorMap{T,S,5,3}) where {T,S} = @tensor tmp′[1,2,3,4,5,6,7,8] * tmp'[6,7,8,1,2,3,4,5]
_inner(tmp′::AbstractTensorMap{T,S,6,3},tmp::AbstractTensorMap{T,S,5,3}) where {T,S} = convert(Array,@tensor s[-1] ≔ tmp′[1,2,3,4,5,-1,6,7,8] * tmp'[6,7,8,1,2,3,4,5])

function _inner(Γ₁::AbstractTensorMap, O::AbstractTensorMap{T,S,2,1}, Γ₂::AbstractTensorMap) where {T,S}
    return convert(Array,@tensor tmp[-1] ≔ Γ₁[1,2,3,5,6] * O[4,-1,3] * Γ₂'[5,6,1,2,4])
end
function _inner(Γ₁::AbstractTensorMap, O::AbstractTensorMap{T,S,1,1}, Γ₂::AbstractTensorMap) where {T,S}
    return @tensor Γ₁[1,2,3,5,6] * O[4,3] * Γ₂'[5,6,1,2,4]
end
function λs(ψ::Dict,Latt::AbstractLattice,i::Int64)
    Lx,Ly = size(Latt)
    ind = Latt[i][2] + [1,1]
    indu = ind[1], mod(ind[2] - 1 + Ly - 1, Ly) + 1
    indl = mod(ind[1] - 1 + Lx - 1, Lx) + 1, ind[2]
    ψ["λr"][ind...], ψ["λu"][ind...], ψ["λu"][indu...], ψ["λr"][indl...]
end


function normalize!(Latt::AbstractLattice, ψ::Dict)
    for i in 1:length(Latt)
        ψ["λu"][i] /= norm(ψ["λu"][i])
        ψ["λr"][i] /= norm(ψ["λr"][i])
        ψ["Γ"][i] /= norm(ψ["Γ"][i])
    end
end

_swap_gate(pspace::ElementarySpace) = permute(id(pspace ⊗ pspace) , (2, 1), (3, 4))

