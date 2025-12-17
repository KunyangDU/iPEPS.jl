initialize!(Map::LatticeMapping, H::Hamiltonian, pspace::ElementarySpace) = initialize!(Map.hamiltonian,H,pspace,Map.nnpairs)

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
    zs = map(x -> x == 0 ? 1 : x, zs)
    H.coordination = Tuple(zs)

    H.partition = let H2nb = collect(keys(H.H2))
        nb0 = sort(filter(x -> x in nbs, H2nb);by = x -> (x[1][1] + x[2][1]))
        nb1 = sort(filter(x -> x ∉ nbs, H2nb);by = x -> (x[1][1] + x[2][1]))
        if isempty(nb1)
            partition(nb0)
        else
            plens = unique(map(y -> length(y[1]),collect(values(H.nnnpath))))
            nbks = [Dict() for _ in eachindex(plens)]
            for (i,pl) in enumerate(plens)
                for n in nb1
                    length(H.nnnpath[n][1]) == pl && (nbks[i][n] = vcat(collect.(H.nnnpath[n])...))
                end
            end
            vcat(partition(nb0), partition.(nbks)...)
        end
    end

    H.pspace = pspace
    return H
end



