

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


function diagm(A::Pair{Int64, Vector{T}}) where T
    L = abs(A.first) + length(A.second)
    B = zeros(T,L,L)
    for i in 1:length(A.second)
        B[(A.first > 0 ? (i,abs(A.first) + i) : (abs(A.first) + i,i))...] = A.second[i]
    end
    return B
end
diagm(A::Vector) = diagm(0 => A)

Base.kron(A::AbstractTensorMap, B::AbstractTensorMap) = @tensor C[a, c; b, d] := A[a; b] * B[c; d]


trivial(::GradedSpace{I, D}) where {I, D} = GradedSpace{I,D}(TensorKit.SortedVectorDict(one(I) => 1), false)
trivial(::ComplexSpace) = ℂ^1

function Base.diff(A::AbstractTensorMap, B::AbstractTensorMap)
    if rank(A) == rank(B)
        return norm(A - B)^2
    elseif rank(A) > rank(B)
        return norm(A - pad(B,codomain(A)))^2
    else
        return norm(B - pad(A,codomain(B)))^2
    end
end


function pad(S::AbstractTensorMap, new_space::VectorSpace)
    # 1. 创建一个全零的新张量，定义在更大的空间上
    S_padded = zeros(eltype(S), new_space, new_space)
    for (c, b) in blocks(S)
        b_padded = block(S_padded, c)
        dims = size(b)
        b_padded[1:dims[1], 1:dims[2]] = b
    end
    return S_padded
end

manualGC() = GC.gc()

_fullize(A::Vector) = unique(vcat(A,_nn_reverse.(A)))

