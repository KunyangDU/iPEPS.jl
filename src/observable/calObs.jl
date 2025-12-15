
function calObs!(O::Observable, ψ::LGState)
    for (i,o) in O.O1
        O.values[i] = real(_calObs1(ψ,o,i))
    end
    for (((i,vi),(j,vj)),o) in O.O2
        O.values[((i,vi),(j,vj))] = real(_calObs2(ψ,o,i,j,ψ.nn2d[(i,vi),(j,vj)]))
    end
    return O
end



function measure(ψ::LGState,H::Hamiltonian)
    E = 0.0
    for ((sitei,sitej),J) in H.H2
        E += _calObs2(ψ,J,sitei,sitej)
    end
    for (i,h) in H.H1 
        E += _calObs1(ψ,h,i)
    end
    return E
end

function _calObs2(ψ::LGState, Os::Vector, i::Int64, j::Int64, ::RIGHT)
    Γl, (λlr, λlu, λld, λll) = ψ[i]
    Γr, (λrr, λru, λrd, λrl) = ψ[j]

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    _,K,_ = kernalize(Γl′,Γr′,RIGHT())
    return map(O -> real(_inner(action(K,O),K)),Os)
end

function _calObs2(ψ::LGState, Os::Vector, i::Int64, j::Int64, ::UP)
    Γu, (λur, λuu, λud, λul) = ψ[j]
    Γd, (λdr, λdu, λdd, λdl) = ψ[i]

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    _,K,_ = kernalize(Γd′,Γu′,UP())
    return map(O -> real(_inner(action(K,O),K)),Os)
end

_calObs2(ψ::LGState,O::AbstractTensorMap, sitei::Tuple, sitej::Tuple) = _calObs2(ψ,[O,],sitei[1],sitej[1],ψ.nn2d[(sitei,sitej)])[1]
_calObs1(ψ::LGState,O::AbstractTensorMap, i::Int64) = _calObs1(ψ,[O,],i)[1]

function _calObs1(ψ::LGState, os::Vector, i::Int64)
    Γ′ = λΓcontract(ψ[i][1],ψ[i][2]...)
    return map(o -> real(_inner(Γ′,o,Γ′)), os)
end
