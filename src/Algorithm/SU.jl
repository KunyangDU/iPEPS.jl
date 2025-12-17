function SU!(ψ::LGState, H::Hamiltonian, algo::SimpleUpdate;
    showperstep::Int64 = 500)
    to = TimerOutput()
    for τ in algo.τs
        tmpto = TimerOutput()
        algo.τ = τ
        E = 0.0
        ΔE = 0.0
        tol = 0.0
        ϵ = 0.0
        for i in 1:algo.N
            # @show space(ψ[1][1])
            ϵ,tol,E′,localto = _SUupdate!(ψ,H,algo;seed = i)
            ΔE,E = E′ - E, E′
            merge!(tmpto,localto)

            # @timeit tmpto "GC" manualGC()

            tol < τ * algo.tol && break
            if i == algo.N
                println("SimpleUpdate update not converged!")
            end
            if mod(i,showperstep) == 0
                show(tmpto;title = "$(i)/$(algo.N)\nτ = $(τ)")
                println("\nE = $(E), ΔE/|E| = $(ΔE / abs(E)), TruncErr = $(ϵ), λErr = $(tol)")
                # print("\n")
            end
        end
        @timeit to "GC" manualGC()
        merge!(to,tmpto)
        show(to;title = "Simple Update\n -> $(τ)")
        println("\nE = $(E), ΔE/|E| = $(ΔE / abs(E)), TruncErr = $(ϵ), λErr = $(tol)")
    end
end


function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::UP, algo::SimpleUpdate)
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

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::RIGHT, algo::SimpleUpdate)
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

# _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::LEFT, algo::SimpleUpdate) = _SUupdate!(ψ , O,j,i,RIGHT(),algo)
# _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::DOWN, algo::SimpleUpdate) = _SUupdate!(ψ , O,j,i,UP(),algo)
_SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::LEFT, algo::SimpleUpdate) = _SUupdate!(ψ , _swap_gate(ψ.pspace) * O * _swap_gate(ψ.pspace),j,i,RIGHT(),algo)
_SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::DOWN, algo::SimpleUpdate) = _SUupdate!(ψ , _swap_gate(ψ.pspace) * O * _swap_gate(ψ.pspace),j,i,UP(),algo)

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, algo::SimpleUpdate)
    to = TimerOutput()
    eO = exp(- algo.τ * O)
    Γ = ψ[i][1]
    @timeit to "action" @tensor Γ′[-1,-2,-3;-4,-5] ≔ Γ[-1,-2,1,-4,-5] * eO[-3,1]
    @timeit to "measure" ΔE = real(λΓcontract(ψ[i]...) |> x -> _inner(x,O,x))
    replace!(ψ,normalize(Γ′),i)
    return ψ, 0.0, 0.0, ΔE, to
end


function _SUupdate!(ψ::LGState, H::Hamiltonian, algo::SimpleUpdate;seed::Int64 = 1)
    ϵ_trunc_tol = 0.0
    ϵ_λ_tol = 0.0
    E = 0.0
    sites1 = collect(keys(H.H1))
    to = TimerOutput()
    
    for (((i,vi),(j,vj)),Heff) in shuffle(collect(H.H2))
        setdiff!(sites1,[i,j])

        @timeit to "build Heff" begin
            if i in keys(H.H1)
                H1 = O1_2_O2_l(H.H1[i],H.pspace) / H.coordination[i]
                Heff += H1
            end

            if j in keys(H.H1)
                H1 = O1_2_O2_r(H.H1[j],H.pspace) / H.coordination[j]
                Heff += H1
            end
        end

        norm(Heff) < 1e-12 && continue
        if haskey(ψ.nn2d,((i,vi),(j,vj)))
            @timeit to "update2!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,Heff,i,j,ψ.nn2d[(i,vi),(j,vj)], algo)
            merge!(to,localto;tree_point = ["update2!"])
        else
            # swap int
            paths = H.nnnpath[((i,vi),(j,vj))]
            path = paths[mod(seed,length(paths)) + 1]
            # path = paths[1]
            @timeit to "swap!" _swap!(ψ,path[1:end-1],algo.trunc)
            (j′,vj′),(i′,vi′) = path[end-1:end]
            @timeit to "update2!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,Heff,i′,j′,ψ.nn2d[(i′,vi′),(j′,vj′)], algo)
            @timeit to "swap!" _swap!(ψ,reverse(path[1:end-1]),algo.trunc)
            merge!(to,localto;tree_point = ["update2!"])
        end

        ϵ_trunc_tol += ϵ_trunc
        ϵ_λ_tol += ϵ_λ
        E += ΔE
    end

    for i in sites1
        norm(H.H1[i]) < 1e-12 && continue
        @timeit to "update1!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,H.H1[i],i, algo)
        merge!(to,localto;tree_point = ["update1!"])
        ϵ_trunc_tol += ϵ_trunc
        ϵ_λ_tol += ϵ_λ
        E += ΔE
    end

    return ϵ_trunc_tol / length(ψ), ϵ_λ_tol / length(ψ), E, to
end