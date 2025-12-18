initialize!(Map::LatticeMapping,ψ::LGState,pspace::ElementarySpace,aspace::ElementarySpace = trivial(pspace)) = initialize!(Map.auxLatt,ψ,pspace,aspace)

# aux Latt: nbs 
function initialize!(Latt::AbstractLattice,ψ::LGState,pspace::ElementarySpace,aspace::ElementarySpace = trivial(pspace))
    nbs = ineighbor(Latt)
    ψ.Γ = [rand(ComplexF64, aspace ⊗ aspace ⊗ pspace, aspace ⊗ aspace) for _ in 1:length(Latt)]
    ψ.λ = [normalize(rand(ComplexF64, aspace, aspace)) for _ in 1:length(nbs)]
    # ψ.nnsites = Tuple(neighborsites_pbc(Latt))
    ψ.pspace = pspace
    ψ.nntable,ψ.nn2d = build_direction_table(Latt)

    nn2λ = Dict()
    for (i,n) in enumerate(nbs)
        nn2λ[n] = i
        nn2λ[_nn_reverse(n)] = i
    end
    # ψ.nn2d = 
    
    λindex = []
    for i in 1:length(Latt)
        push!(λindex,_λindex(ψ,i,nn2λ))
    end
    ψ.λindex = Tuple(λindex)
    _check_λ_index(ψ)

    return ψ
end
