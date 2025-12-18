
function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::UP, algo::SimpleUpdate{FullSU})
    to = TimerOutput()
    Γu, λur, λuu, λud, λul = ψ[j]
    Γd, λdr, λdu, λdd, λdl = ψ[i]

    @timeit to "λ-Γ contract" Γu′ = λΓcontract(Γu, λur, λuu, sqrt(λud), λul)
    @timeit to "λ-Γ contract" Γd′ = λΓcontract(Γd, λdr, sqrt(λdu), λdd, λdl)

    @timeit to "kernalize" DΓ,K,UΓ = kernalize(Γd′,Γu′,UP())

    @timeit to "action" C = action(K,exp(- algo.τ * O))
    @timeit to "measure" ΔE = real(_inner(action(K,O),K))
    @timeit to "svd" U,Λ,V,ϵ_trunc = tsvd(C,(1,2),(3,4);trunc = algo.trunc)

    @timeit to "dekernalize" Γd′ = dekernalize(DΓ,V,UP())
    @timeit to "dekernalize" Γu′ = dekernalize(UΓ,U,DOWN())

    Λ = normalize(Λ)
    ϵ_λ = diff(ψ[i][3],Λ)

    replace!(ψ,Λ,i,UP())
    replace!(ψ,invu(Γu′,λur, λuu, λul),j)
    replace!(ψ,invd(Γd′,λdr, λdd, λdl),i)

    return ψ, ϵ_trunc^2, ϵ_λ, ΔE, to
end

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::RIGHT, algo::SimpleUpdate{FullSU})
    to = TimerOutput()
    Γl, λlr, λlu, λld, λll = ψ[i]
    Γr, λrr, λru, λrd, λrl = ψ[j]

    @timeit to "λ-Γ contract" Γl′ = λΓcontract(Γl, sqrt(λlr), λlu, λld, λll)
    @timeit to "λ-Γ contract" Γr′ = λΓcontract(Γr, λrr, λru, λrd, sqrt(λrl))

    @timeit to "kernalize" LΓ,K,RΓ = kernalize(Γl′,Γr′,RIGHT())

    @timeit to "action" C = action(K,exp(- algo.τ * O))
    @timeit to "measure" ΔE = real(_inner(action(K,O),K))
    @timeit to "svd" U,Λ,V,ϵ_trunc = tsvd(C,(1,2),(3,4);trunc = algo.trunc)
    
    @timeit to "dekernalize" Γl′ = dekernalize(LΓ,V,RIGHT())
    @timeit to "dekernalize" Γr′ = dekernalize(RΓ,U,LEFT())

    Λ = normalize(Λ)
    ϵ_λ = diff(ψ[i][2], Λ)

    replace!(ψ,Λ,i,RIGHT())
    replace!(ψ,invl(Γl′,λlu, λll, λld),i)
    replace!(ψ,invr(Γr′,λrr, λru, λrd),j)

    return ψ, ϵ_trunc^2, ϵ_λ, ΔE, to
end


function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, algo::SimpleUpdate{FullSU})
    to = TimerOutput()
    eO = exp(- algo.τ * O)
    Γ = ψ[i][1]
    @timeit to "action" @tensor Γ′[-1,-2,-3;-4,-5] ≔ Γ[-1,-2,1,-4,-5] * eO[-3,1]
    @timeit to "measure" ΔE = real(λΓcontract(ψ[i]...) |> x -> _inner(x,O,x))
    replace!(ψ,normalize(Γ′),i)
    return ψ, 0.0, 0.0, ΔE, to
end