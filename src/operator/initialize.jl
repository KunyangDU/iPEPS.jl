function initialize!(Latt::AbstractLattice, H::Hamiltonian, pspace::ElementarySpace,nbs::Vector = ineighbor(Latt))

    # nnnnb = filter(x -> x ∉ nbs, collect(keys(H.H2)))
    # osites = [(i,[0,0]) for i in 1:length(Latt)]
    # its = map(x -> map(z -> z[2], filter(y -> y[1] == x,nnnnb)),osites)
    # paths = map(x -> findpath(Latt,osites[x],its[x]), 1:length(Latt))

    # for i in 1:length(Latt),j in eachindex(its[i])
    #     H.nnnpath[((i,[0,0]),its[j])] = Tuple(paths[i][j])
    # end
    zs = zeros(Int64,length(Latt))
    for nb in keys(H.H2)
        zs[nb[1][1]] += 1
        zs[nb[2][1]] += 1
        nb ∈ nbs && continue
        isnothing(H.nnnpath) && (H.nnnpath = Dict{Tuple,Tuple}())
        H.nnnpath[nb] = Tuple(findpath(Latt,nb...))
    end

    H.coordination = Tuple(zs)

    H.pspace = pspace
    return H
end
