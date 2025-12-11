
function measure2(ψ::Dict,H::Dict)
    E = 0.0
    for ((i,j),v) in H["sites2"]
        E += _calObs2(ψ,H["H2"],Latt,i,j,v)
    end
    return E
end

