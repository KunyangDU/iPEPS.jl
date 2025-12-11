function _calObs2(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::RIGHT)
    λlr, λlu, λld, λll= λs(ψ,Latt,i)
    λrr, λru, λrd, λrl= λs(ψ,Latt,j)
    Γl = ψ["Γ"][(Latt[i][2] + [1,1])...]
    Γr = ψ["Γ"][(Latt[j][2] + [1,1])...]
    @assert λlr == λrl

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    tmp = Γcontractlr(Γl′,Γr′)
    tmp′ = actionlr(tmp,O)

    return real(_inner(tmp′,tmp))
end

function _calObs2(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::UP)
    λur, λuu, λud, λul= λs(ψ,Latt,j)
    λdr, λdu, λdd, λdl= λs(ψ,Latt,i)
    Γu = ψ["Γ"][(Latt[j][2] + [1,1])...]
    Γd = ψ["Γ"][(Latt[i][2] + [1,1])...]
    @assert λud ≈ λdu

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    tmp = Γcontractud(Γd′, Γu′)
    tmp′ = actionud(tmp, O)

    return real(_inner(tmp′,tmp))
end


function _calObs2(ψ::Dict,O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, v::Vector)
    R = Latt[j][2] - Latt[i][2] + v
    if R == [1,0]
        return _calObs2(ψ,O,Latt,i,j,RIGHT())
    elseif R == [0,1]
        return _calObs2(ψ,O,Latt,i,j,UP())
    end
end