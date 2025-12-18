using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 2
Ly = 2
Latt = ZZPeriHoneycomb(Lx,Ly)
@save "Honeycomb/data/Latt_$(Lx)x$(Ly).jld2" Latt
Map = ZZPeriHoneycombMapping(Latt)
params = (J1 = 0.0, J3 = 0.0, h = 0.0)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
    addIntr2!(H, ineighbor(Latt;level = 3), LocalSpace.SJ(params.J3 * diagm(ones(3))))
    # addIntr1!(H, 1:length(Latt), - params.h * LocalSpace.Sh([0,0,1]))
    addIntr1!(H, 1, - 100 * LocalSpace.Sh([0,0,1]))
    @time initialize!(Map,H,ℂ^2)
end

ψ = LGState(Map)
initialize!(Map,ψ,ℂ^2)

D = 4

sualgo = SimpleUpdate(
    DynamicSU(3),
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    1000,
    [1,0.1,0.01,0.001],
    0.0,
    0.0
)

SU!(ψ,H,sualgo)

# H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
#     addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
#     addIntr2!(H, ineighbor(Latt;level = 3), LocalSpace.SJ(params.J3 * diagm(ones(3))))
#     initialize!(Map,H,ℂ^2)
# end

# sualgo.τs = [0.01,0.001]
# SU!(ψ,H,sualgo)


O = let obs = Observable(), LocalSpace = TrivialSpinOneHalf
    addObs1!(obs,1:length(Latt),LocalSpace.Sx)
    addObs1!(obs,1:length(Latt),LocalSpace.Sy)
    addObs1!(obs,1:length(Latt),LocalSpace.Sz)
    # addObs2!(obs,ineighbor(Latt),LocalSpace.SxSx)
    # addObs2!(obs,ineighbor(Latt),LocalSpace.SySy)
    # addObs2!(obs,ineighbor(Latt),LocalSpace.SzSz)
    initialize!(Latt,obs)
end

calObs!(O,ψ)
data = Dict(
    "Obs" => O.values,
    "E" => measure(ψ,H,sualgo),
)

@save "Honeycomb/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@save "Honeycomb/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

data["E"]
