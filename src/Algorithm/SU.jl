
function _SUupdate!(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::RIGHT, SUalgo::Dict)
    λlr, λlu, λld, λll= λs(ψ,Latt,i)
    λrr, λru, λrd, λrl= λs(ψ,Latt,j)
    Γl = ψ["Γ"][(Latt[i][2] + [1,1])...]
    Γr = ψ["Γ"][(Latt[j][2] + [1,1])...]
    @assert λlr == λrl

    Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    tmp′ = actionlr(Γcontractlr(Γl′,Γr′),O) |> x -> x + SUalgo["noise"] * rand(space(x))

    Γr′,Λ,Γl′,ϵ_trunc = tsvd(tmp′,(1,2,4,6),(3,5,7,8);trunc = truncbelow(SUalgo["ϵ"]) & truncdim(SUalgo["D"]))

    Γl′ = permute(Γl′,(1,2,3),(4,5))
    Γr′ = permute(Γr′,(1,2,3),(4,5))
    Λ = normalize(Λ)

    ϵ_λ = diff(ψ["λr"][(Latt[i][2] + [1,1])...], Λ)
    ψ["λr"][(Latt[i][2] + [1,1])...] = Λ


    ψ["Γ"][(Latt[i][2] + [1,1])...] = invl(Γl′,λlu, λll, λld)
    ψ["Γ"][(Latt[j][2] + [1,1])...] = invr(Γr′,λrr, λru, λrd)
    return ψ, ϵ_trunc, ϵ_λ
end

function _SUupdate!(ψ::Dict, O::AbstractTensorMap,Latt::AbstractLattice, i::Int64, j::Int64, ::UP, SUalgo::Dict)
    λur, λuu, λud, λul= λs(ψ,Latt,j)
    λdr, λdu, λdd, λdl= λs(ψ,Latt,i)
    Γu = ψ["Γ"][(Latt[j][2] + [1,1])...]
    Γd = ψ["Γ"][(Latt[i][2] + [1,1])...]
    @assert λud == λdu

    Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    tmp′ = actionud(Γcontractud(Γd′,Γu′), O) |> x -> x + SUalgo["noise"] * rand(space(x))

    Γu′,Λ,Γd′,ϵ_trunc = tsvd(tmp′,(2,3,5,8),(1,4,6,7);trunc = truncbelow(SUalgo["ϵ"]) & truncdim(SUalgo["D"]))
    Γu′ = permute(Γu′,(1,2,3),(5,4))
    Γd′ = permute(Γd′,(2,1,3),(4,5))
    Λ = normalize(Λ)
    ϵ_λ = diff(ψ["λu"][(Latt[i][2] + [1,1])...],Λ)

    ψ["λu"][(Latt[i][2] + [1,1])...] = Λ
    ψ["Γ"][(Latt[j][2] + [1,1])...] = invu(Γu′,λur, λuu, λul)
    ψ["Γ"][(Latt[i][2] + [1,1])...] = invd(Γd′,λdr, λdd, λdl)
    return ψ, ϵ_trunc, ϵ_λ
end


function _SUupdate!(ψ::Dict,H::Dict,Latt::AbstractLattice, SUalgo::Dict;pspace = ℂ^2)
    ϵ_trunc_tol = 0.0
    ϵ_λ_tol = 0.0
    for (ind,((i,j),v)) in enumerate(H["sites2"])
        R = Latt[j][2] - Latt[i][2] + v
        
        Heff = H["H2"]
        if i in H["sites1"]
            H1 = O1_2_O2_r(H["H1"],pspace) / H["sites1nb"][i]
            Heff += H1
        end

        if j in H["sites1"]
            H1 = O1_2_O2_l(H["H1"],pspace) / H["sites1nb"][j]
            Heff += H1
        end

        O = exp(- SUalgo["τ"] * Heff)
        if R == [1,0]
            _,ϵ_trunc,ϵ_λ = _SUupdate!(ψ,O,Latt,i,j,RIGHT(),SUalgo)
        elseif R == [0,1]
            _,ϵ_trunc,ϵ_λ = _SUupdate!(ψ,O,Latt,i,j,UP(),SUalgo)
        end
        ϵ_trunc_tol += ϵ_trunc
        ϵ_λ_tol += ϵ_λ
    end
    return ϵ_trunc_tol / 2length(ψ["Γ"]), ϵ_λ_tol / 2length(ψ["Γ"])
end

function SUupdate!(ψ::Dict,H::Dict,Latt::AbstractLattice, SUalgo::Dict)
    for τ in SUalgo["τs"]
        SUalgo["τ"] = τ
        for i in 1:SUalgo["N"]
            SUalgo["noise"] = 0.0
            ϵ,tol = _SUupdate!(ψ,H,Latt,SUalgo)
            tol < τ * SUalgo["tol"] && break
            if i == SUalgo["N"]
                println("SU update not converged!")
            end
        end
    end
end
