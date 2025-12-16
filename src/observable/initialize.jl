initialize!(Map::LatticeMapping, O::Observable) = initialize!(Map.hamiltonian,O,Map.nnpairs)

# nbs -> aux Latt 
function initialize!(Latt::AbstractLattice, O::Observable, nbs::Vector = ineighbor(Latt))
    for nb in keys(O.O2)
        nb ∈ nbs && continue
        isnothing(O.nnnpath) && (O.nnnpath = Dict{Tuple,Tuple}())
        O.nnnpath[nb] = Tuple(findpath(Latt,nb...))
    end

    return O
end