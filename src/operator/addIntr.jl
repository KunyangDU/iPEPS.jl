
function addIntr2!(H::Hamiltonian,nb::Tuple,J::TensorMap)
    norm(J) < 1e-12 && return H
    if haskey(H.H2,nb)
        H.H2[nb] += J
    else
        H.H2[nb] = J
    end
    return H
end

function addIntr1!(H::Hamiltonian,i::Int64,h::TensorMap)
    norm(h) < 1e-12 && return H
    if haskey(H.H1,i)
        H.H1[i] += h
    else
        H.H1[i] = h
    end
    return H
end

function addIntr2!(H::Hamiltonian,nbs::Vector,J::TensorMap)
    norm(J) < 1e-12 && return H
    for nb in nbs
        addIntr2!(H,nb,J)
    end
    return H
end

function addIntr1!(H::Hamiltonian,sites::Union{UnitRange,Vector},h::TensorMap)
    norm(h) < 1e-12 && return H
    for i in sites
        addIntr1!(H,i,h)
    end
    return H
end