using TensorKit,Random
include("../src/iPEPS.jl")


# Lx = 2
# Ly = 2
# Latt = PeriSqua(Lx,Ly)


# phyLatt = ZZPeriHoneycomb(2,2)
# Map = ZZPeriHoneycombMapping(phyLatt)
# auxLatt = Map.auxLatt


phyLatt = XCPeriTria(3,3)
Map = XCPeriTriaMapping(phyLatt)
auxLatt = Map.auxLatt

banlist = setdiff(_fullize(ineighbor(auxLatt)),_fullize(ineighbor(phyLatt)))
findpath(auxLatt,(2,[0,0]),[(4,[0,0]),],banlist)


# neighborsites_pbc(Map.auxLatt;banlist = banlist)
# setdiff(neighbor_pbc(Latt;level = 1, issort = false, ordered = false),banlist)
# banlist
# @load "Heisenberg/data/Latt_$(Lx)x$(Ly).jld2" Latt
# D = 2
# params = (J = 1.0, h = 0.0)
# @load "Heisenberg/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ


# H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
#     addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
#     addIntr2!(H, ineighbor(Latt;level = 2), LocalSpace.SJ(params.J2 * diagm(ones(3))))
#     addIntr1!(H,1,LocalSpace.Sh(-[0,0,100]))
#     initialize!(Latt,H,ℂ^2)
# end

# H.nnnpath[((1, [0, 0]), (4, [0, 0]))]
