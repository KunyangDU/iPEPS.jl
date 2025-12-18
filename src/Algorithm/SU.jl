function SU!(ψ::LGState, H::Hamiltonian, algo::SimpleUpdate{Sch};
    showperstep::Int64 = 500) where Sch
    to = TimerOutput()
    if Sch <: FastSU || (Sch <:DynamicSU && algo.scheme.N > 0 )
        @timeit to "merge_λ" merge_λ!(ψ)
    end

    for (iτ,τ) in enumerate(algo.τs)
        tmpto = TimerOutput()
        algo.τ = τ
        E = 0.0
        ΔE = 0.0
        tol = 0.0
        ϵ = 0.0
        Sch <:DynamicSU && (algo.scheme.count += 1)
        
        for i in 1:algo.N
            ϵ,tol,E′,localto = _SUupdate!(ψ,H,algo;seed = i)
            ΔE,E = E′ - E, E′
            merge!(tmpto,localto)

            tol < τ * algo.tol && break
            if i == algo.N
                println("SimpleUpdate update not converged!")
            end
            if mod(i,showperstep) == 0
                show(tmpto;title = "τ = $(τ) - $(i)/$(algo.N)")
                println("\nE = $(E), ΔE/|E| = $(ΔE / abs(E)), TruncErr = $(ϵ), λErr = $(tol)")
            end
        end
        
        @timeit to "GC" manualGC()
        (Sch <:DynamicSU && iτ == algo.scheme.N) && (@timeit tmpto "dismerge_λ" dismerge_λ!(ψ))
        merge!(to,tmpto)
        show(to;title = "Simple Update")
        println("\nE = $(E), ΔE/|E| = $(ΔE / abs(E)), TruncErr = $(ϵ), λErr = $(tol)")
    end
    Sch <: FastSU && dismerge_λ!(ψ)
    return ψ
end

# _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::LEFT, algo::SimpleUpdate) = _SUupdate!(ψ , O,j,i,RIGHT(),algo)
# _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::DOWN, algo::SimpleUpdate) = _SUupdate!(ψ , O,j,i,UP(),algo)
_SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::LEFT, algo::SimpleUpdate{T}) where T <: Union{FastSU,FullSU} = _SUupdate!(ψ , _swap_gate(ψ.pspace) * O * _swap_gate(ψ.pspace),j,i,RIGHT(),algo)
_SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, ::DOWN, algo::SimpleUpdate{T}) where T <: Union{FastSU,FullSU} = _SUupdate!(ψ , _swap_gate(ψ.pspace) * O * _swap_gate(ψ.pspace),j,i,UP(),algo)

function _SUupdate!(ψ::LGState, H::Hamiltonian, algo::SimpleUpdate;seed::Int64 = 1)
    ϵ_trunc_tol = 0.0
    ϵ_λ_tol = 0.0
    E = 0.0
    sites1 = collect(keys(H.H1))
    to = TimerOutput()

    Nthr = get_num_threads_julia()
    if Nthr > 1
        for items in shuffle(H.partition)
            n_blas = min(max(1, div(Nthr, length(items))), 8)
            BLAS.set_num_threads(n_blas)

            thread_buffers = [[TimerOutput(),0.0,0.0,0.0,[]] for _ in 1:Nthr]

            Threads.@threads for ((i,vi),(j,vj)) in items
                tid = Threads.threadid()
                thrto = thread_buffers[tid][1]
                Heff = H.H2[((i,vi),(j,vj))]

                @timeit thrto "build Heff" begin
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
                    @timeit thrto "update2!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,Heff,i,j,ψ.nn2d[(i,vi),(j,vj)], algo)
                    merge!(thrto,localto;tree_point = ["update2!"])
                else
                    # swap int
                    paths = H.nnnpath[((i,vi),(j,vj))]
                    path = paths[mod(seed,length(paths)) + 1]
                    # path = paths[1]
                    @timeit thrto "swap!" _swap!(ψ,path[1:end-1],algo)
                    (j′,vj′),(i′,vi′) = path[end-1:end]
                    @timeit thrto "update2!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,Heff,i′,j′,ψ.nn2d[(i′,vi′),(j′,vj′)], algo)
                    @timeit thrto "swap!" _swap!(ψ,reverse(path[1:end-1]),algo)
                    merge!(thrto,localto;tree_point = ["update2!"])
                end

                thread_buffers[tid][2] += ϵ_trunc
                thread_buffers[tid][3] += ϵ_λ
                thread_buffers[tid][4] += ΔE
                thread_buffers[tid][5] = [i,j]
                # merge!()
            end
            for bf in thread_buffers
                merge!(to,bf[1])
                ϵ_trunc_tol += bf[2]
                ϵ_λ_tol += bf[3]
                E += bf[4]
                setdiff!(sites1,bf[5])
            end
        end

    else
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
                @timeit to "swap!" _swap!(ψ,path[1:end-1],algo)
                (j′,vj′),(i′,vi′) = path[end-1:end]
                @timeit to "update2!" _,ϵ_trunc,ϵ_λ,ΔE,localto = _SUupdate!(ψ,Heff,i′,j′,ψ.nn2d[(i′,vi′),(j′,vj′)], algo)
                @timeit to "swap!" _swap!(ψ,reverse(path[1:end-1]),algo)
                merge!(to,localto;tree_point = ["update2!"])
            end

            ϵ_trunc_tol += ϵ_trunc
            ϵ_λ_tol += ϵ_λ
            E += ΔE
        end
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