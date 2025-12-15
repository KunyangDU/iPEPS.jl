
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