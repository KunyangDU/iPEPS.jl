_ZZHC_index(phyLatt::AbstractLattice) = Tuple(sort(1:length(phyLatt);by = x -> coordinate(phyLatt,x)[2]))
_XCTria_index(phyLatt::AbstractLattice) = Tuple(1:length(phyLatt))
mutable struct LatticeMapping <: AbstractMapping
    phyLatt::AbstractLattice
    auxLatt::AbstractLattice
    state::AbstractLattice
    hamiltonian::AbstractLattice
    nnpairs::Vector
end

function ZZPeriHoneycombMapping(phyLatt::AbstractLattice)
    auxLatt = PeriSqua((2 .* size(phyLatt))...,_ZZHC_index(phyLatt))
    return LatticeMapping(phyLatt,auxLatt,auxLatt,phyLatt,_fullize(ineighbor(auxLatt)))
end

function XCPeriTriaMapping(phyLatt::AbstractLattice)
    auxLatt = PeriSqua((size(phyLatt))...,_XCTria_index(phyLatt))
    return LatticeMapping(phyLatt,auxLatt,auxLatt,auxLatt,_fullize(ineighbor(auxLatt)))
end

function PeriSquaMapping(phyLatt::AbstractLattice)
    LatticeMapping(phyLatt,phyLatt,phyLatt,phyLatt,_fullize(ineighbor(phyLatt)))
end
