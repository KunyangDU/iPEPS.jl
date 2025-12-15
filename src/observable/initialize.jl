
function initialize!(Latt::AbstractLattice, O::Observable)

    nbs = ineighbor(Latt)

    for nb in keys(O.O2)
        nb ∈ nbs && continue
        isnothing(O.nnnpath) && (O.nnnpath = Dict{Tuple,Tuple}())
        O.nnnpath[nb] = Tuple(findpath(Latt,nb...))
    end

    return O
end