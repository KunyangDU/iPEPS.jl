initialize!(Map::LatticeMapping, O::Observable) = initialize!(Map.auxLatt,O,Map.nnpairs,Map.banbonds)

# nbs -> aux Latt 
function initialize!(Latt::AbstractLattice, O::Observable, nbs::Vector = ineighbor(Latt),banbonds::Vector = [])
    for nb in keys(O.O2)
        nb ∈ nbs && continue
        isnothing(O.nnnpath) && (O.nnnpath = Dict{Tuple,Tuple}())
        O.nnnpath[nb] = Tuple(findpath(Latt,nb...,banbonds))
    end

    return O
end

