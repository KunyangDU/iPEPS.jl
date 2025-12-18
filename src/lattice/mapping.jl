_ZZHC_index(phyLatt::AbstractLattice) = Tuple(sort(1:length(phyLatt);by = x -> coordinate(phyLatt,x)[2]))
_XCTria_index(phyLatt::AbstractLattice) = Tuple(1:length(phyLatt))
function _Kagome_index(phyLatt::AbstractLattice)
    totalsite = convert(Vector{Any},sort(1:length(phyLatt);by = x -> coordinate(phyLatt,x)[2]))
    othersite = collect(length(phyLatt) + 1:length(phyLatt) + *(size(phyLatt)...))
    L = length(othersite)
    # end
    for (ind,i) in enumerate(totalsite)
        isempty(othersite) && break
        i > length(Latt) && continue 
        Latt[i][1] == 3 && (totalsite[ind] = [i,popfirst!(othersite)])
    end
    return Tuple(vcat(totalsite...))
end

mutable struct LatticeMapping <: AbstractMapping
    phyLatt::AbstractLattice
    auxLatt::AbstractLattice
    # state::AbstractLattice
    # hamiltonian::AbstractLattice
    nnpairs::Vector
    banbonds::Vector
end

function ZZPeriHoneycombMapping(phyLatt::AbstractLattice)
    auxLatt = PeriSqua((2 .* size(phyLatt))...,_ZZHC_index(phyLatt))
    aux_nnpairs = _fullize(ineighbor(auxLatt))
    phy_nnpairs = _fullize(ineighbor(phyLatt))
    return LatticeMapping(phyLatt,auxLatt,intersect(aux_nnpairs, phy_nnpairs),setdiff(aux_nnpairs,phy_nnpairs))
end

function XCPeriTriaMapping(phyLatt::AbstractLattice)
    auxLatt = PeriSqua((size(phyLatt))...,_XCTria_index(phyLatt))
    aux_nnpairs = _fullize(ineighbor(auxLatt))
    phy_nnpairs = _fullize(ineighbor(phyLatt))
    return LatticeMapping(phyLatt,auxLatt,intersect(aux_nnpairs, phy_nnpairs),setdiff(aux_nnpairs,phy_nnpairs))

end

function PeriKagomeMapping(phyLatt::AbstractLattice)
    auxLatt = PeriSqua((2 .* size(phyLatt))...,_Kagome_index(phyLatt))
    aux_nnpairs = _fullize(ineighbor(auxLatt))
    phy_nnpairs = _fullize(ineighbor(phyLatt))
    return LatticeMapping(phyLatt,auxLatt,intersect(aux_nnpairs, phy_nnpairs),setdiff(aux_nnpairs,phy_nnpairs))

end

function PeriSquaMapping(phyLatt::AbstractLattice)
    return LatticeMapping(phyLatt,phyLatt,_fullize(ineighbor(phyLatt)),[])
end
