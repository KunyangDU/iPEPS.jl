using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 4
Ly = 4
Latt = PeriSqua(Lx,Ly)
@save "Square_SU2/data/Latt_$(Lx)x$(Ly).jld2" Latt
Map = PeriSquaMapping(Latt)
params = (J1 = 1.0, J2 = 0.0, h = 0.0)
LocalSpace = SU₂Spin
H = let H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), params.J1 * LocalSpace.SS)
    addIntr2!(H, ineighbor(Latt;level = 2), params.J2 * LocalSpace.SS)
    # addIntr1!(H,1,LocalSpace.Sh(-[0,0,10000]))
    initialize!(Map,H,LocalSpace.pspace)
end

ψ = LGState(Map)
initialize!(Map,ψ,LocalSpace.pspace,Rep[SU₂](i => 1 for i in 0:1//2:2))

D = 4

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    1000,
    [0.1,],
    0.0,
    0.0
)

# ψ[1][1]

SU!(ψ,H,sualgo)

# H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
#     addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
#     addIntr2!(H, ineighbor(Latt;level = 2), LocalSpace.SJ(params.J2 * diagm(ones(3))))
#     initialize!(Map,H,ℂ^2)
# end

# sualgo.τs = [0.01,0.001]
# SU!(ψ,H,sualgo)


O = let obs = Observable()
    addObs2!(obs,ineighbor(Latt),LocalSpace.SS)
    initialize!(Map,obs)
end

calObs!(O,ψ)
data = Dict(
    "Obs" => O.values,
    "E" => measure(ψ,H,sualgo.trunc),
)

@save "Square_SU2/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@save "Square_SU2/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

@show data["E"]
data["Obs"]
