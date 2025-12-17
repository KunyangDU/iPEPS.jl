using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 3
Ly = 3
Latt = XCPeriTria(Lx,Ly)
@save "Triangular/data/Latt_$(Lx)x$(Ly).jld2" Latt
Map = XCPeriTriaMapping(Latt)

D = 5

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    5000,
    [0.1,0.01,0.001],
    0.0,
    0.0
)

# for h in 4.3:0.1:4.7
params = (J1 = 1.0, J2 = 0.0, h = 0.0)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
    addIntr2!(H, ineighbor(Latt;level = 2), LocalSpace.SJ(params.J2 * diagm(ones(3))))
    addIntr1!(H,1:length(Latt),params.h * LocalSpace.Sh(-[0,0,1]))
    initialize!(Map,H,ℂ^2)
end

ψ = LGState(Map)
initialize!(Map,ψ,ℂ^2)

H.partition
# SU!(ψ,H,sualgo)

# H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
#     addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
#     addIntr2!(H, ineighbor(Latt;level = 2), LocalSpace.SJ(params.J2 * diagm(ones(3))))
#     initialize!(Map,H,ℂ^2)
# end

# sualgo.τs = [0.01,0.001]
# SU!(ψ,H,sualgo)


# O = let obs = Observable(), LocalSpace = TrivialSpinOneHalf
#     addObs1!(obs,1:length(Latt),LocalSpace.Sx)
#     addObs1!(obs,1:length(Latt),LocalSpace.Sy)
#     addObs1!(obs,1:length(Latt),LocalSpace.Sz)
#     # addObs2!(obs,ineighbor(Latt),LocalSpace.SxSx)
#     # addObs2!(obs,ineighbor(Latt),LocalSpace.SySy)
#     # addObs2!(obs,ineighbor(Latt),LocalSpace.SzSz)
#     initialize!(Latt,obs)
# end

# calObs!(O,ψ)
# data = Dict(
#     "Obs" => O.values,
#     "E" => measure(ψ,H,sualgo.trunc),
# )

# @save "Triangular/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
# @save "Triangular/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

# data["E"]
# end
