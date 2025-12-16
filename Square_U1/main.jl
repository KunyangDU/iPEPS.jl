using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 4
Ly = 4
Latt = PeriSqua(Lx,Ly)
@save "Square_U1/data/Latt_$(Lx)x$(Ly).jld2" Latt
Map = PeriSquaMapping(Latt)
params = (J1 = 1.0, J2 = 0.75, h = 0.0)
LocalSpace = U₁Spin
H = let H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), params.J1 * LocalSpace.SS)
    addIntr2!(H, ineighbor(Latt;level = 2), params.J2 * LocalSpace.SS)
    # addIntr1!(H,1,- 100 * LocalSpace.Sz)
    initialize!(Map,H,LocalSpace.pspace)
end

ψ = LGState(Map)
initialize!(Map,ψ,LocalSpace.pspace,Rep[U₁](i => 1 for i in -5/2:1/2:5/2))

D = 4

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    1000,
    [0.1,0.01,0.001],
    0.0,
    0.0
)


SU!(ψ,H,sualgo)


O = let obs = Observable(), LocalSpace = U₁Spin
    addObs1!(obs,1:length(Latt),LocalSpace.Sz)
    addObs2!(obs,ineighbor(Latt),LocalSpace.SzSz)
    initialize!(Map,obs)
end

calObs!(O,ψ)
data = Dict(
    "Obs" => O.values,
    "E" => measure(ψ,H,sualgo.trunc),
)

@save "Square_U1/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@save "Square_U1/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

@show data["E"]
data["Obs"]
