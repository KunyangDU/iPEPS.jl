
function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::UP, algo::SimpleUpdate{FastSU})
    to = TimerOutput()
    Γu, _, _, λud, _ = ψ[j]
    Γd, _, λdu, _, _ = ψ[i]

    isλud = inv(sqrt(λud))
    isλdu = inv(sqrt(λdu))
    @timeit to "λ-Γ contract" (@tensor Γu′[-1,-2,-3;-4,-5] ≔ Γu[-1,-2,-3,1,-5] * isλud[1,-4])
    @timeit to "λ-Γ contract" (@tensor Γd′[-1,-2,-3;-4,-5] ≔ Γd[-1,1,-3,-4,-5] * isλdu[-2,1])

    @timeit to "kernalize" DΓ,K,UΓ = kernalize(Γd′,Γu′,UP())

    @timeit to "action" C = action(K,exp(- algo.τ * O))
    @timeit to "measure" ΔE = real(_inner(action(K,O),K))
    @timeit to "svd" U,Λ,V,ϵ_trunc = tsvd(C,(1,2),(3,4);trunc = algo.trunc)

    @timeit to "dekernalize" Γd′ = dekernalize(DΓ,V,UP())
    @timeit to "dekernalize" Γu′ = dekernalize(UΓ,U,DOWN())

    Λ = normalize(Λ)
    ϵ_λ = diff(ψ[i][3],Λ)

    @timeit to "λ-Γ contract" (@tensor Γu′[-1,-2,-3;-4,-5] = Γu′[-1,-2,-3,1,-5] * Λ[1,-4])
    @timeit to "λ-Γ contract" (@tensor Γd′[-1,-2,-3;-4,-5] = Γd′[-1,1,-3,-4,-5] * Λ[-2,1])

    replace!(ψ,Λ,i,UP())
    replace!(ψ,Γu′,j)
    replace!(ψ,Γd′,i)

    return ψ, ϵ_trunc^2, ϵ_λ, ΔE, to
end

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::RIGHT, algo::SimpleUpdate{FastSU})
    to = TimerOutput()
    Γl, λlr, _, _, _ = ψ[i]
    Γr, _, _, _, λrl = ψ[j]

    isλlr = inv(sqrt(λlr))
    isλrl = inv(sqrt(λrl))
    @timeit to "λ-Γ contract" (@tensor Γl′[-1,-2,-3;-4,-5] ≔ Γl[1,-2,-3,-4,-5] * isλlr[-1,1])
    @timeit to "λ-Γ contract" (@tensor Γr′[-1,-2,-3;-4,-5] ≔ Γr[-1,-2,-3,-4,1] * isλrl[1,-5])

    @timeit to "kernalize" LΓ,K,RΓ = kernalize(Γl′,Γr′,RIGHT())

    @timeit to "action" C = action(K,exp(- algo.τ * O))
    @timeit to "measure" ΔE = real(_inner(action(K,O),K))
    @timeit to "svd" U,Λ,V,ϵ_trunc = tsvd(C,(1,2),(3,4);trunc = algo.trunc)
    
    @timeit to "dekernalize" Γl′ = dekernalize(LΓ,V,RIGHT())
    @timeit to "dekernalize" Γr′ = dekernalize(RΓ,U,LEFT())

    Λ = normalize(Λ)
    ϵ_λ = diff(ψ[i][2], Λ)

    @timeit to "λ-Γ contract" (@tensor Γl′[-1,-2,-3;-4,-5] = Γl′[1,-2,-3,-4,-5] * Λ[-1,1])
    @timeit to "λ-Γ contract" (@tensor Γr′[-1,-2,-3;-4,-5] = Γr′[-1,-2,-3,-4,1] * Λ[1,-5])

    replace!(ψ,Λ,i,RIGHT())
    replace!(ψ,Γl′,i)
    replace!(ψ,Γr′,j)

    return ψ, ϵ_trunc^2, ϵ_λ, ΔE, to
end


function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, algo::SimpleUpdate{FastSU})
    to = TimerOutput()
    eO = exp(- algo.τ * O)
    Γ = ψ[i][1]
    @timeit to "action" @tensor Γ′[-1,-2,-3;-4,-5] ≔ Γ[-1,-2,1,-4,-5] * eO[-3,1]
    @timeit to "measure" ΔE = real(ψ[i][1] |> x -> _inner(x,O,x))
    replace!(ψ,normalize(Γ′),i)
    return ψ, 0.0, 0.0, ΔE, to
end