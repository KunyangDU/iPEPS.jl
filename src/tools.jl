

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

