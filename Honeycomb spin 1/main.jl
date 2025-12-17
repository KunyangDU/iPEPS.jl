using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 3
Ly = 1
Latt = ZZPeriHoneycomb(Lx,Ly)
@save "Honeycomb spin 1/data/Latt_$(Lx)x$(Ly).jld2" Latt
Map = ZZPeriHoneycombMapping(Latt)


D = 6

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    2000,
    [0.1,0.01,],
    0.0,
    0.0
)

for h in 2.4
params = (J1 = -1.0, J3 = 1.0, K = 0.6, D = -3.0, h = h, θ = pi/18)
LocalSpace = TrivialSpinOne
xb,yb,zb = getxyzbonds(Latt)
H = let H = Hamiltonian(),proj = - sin(params.θ) * [1,1,-2] / sqrt(2) + cos(params.θ) * [1,1,1]/sqrt(3)
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
    addIntr2!(H, xb, params.K * LocalSpace.SxSx)
    addIntr2!(H, yb, params.K * LocalSpace.SySy)
    addIntr2!(H, zb, params.K * LocalSpace.SzSz)
    addIntr2!(H, ineighbor(Latt;level = 3), LocalSpace.SJ(params.J3 * diagm(ones(3))))
    addIntr1!(H, 1:length(Latt), params.D * (LocalSpace.Sh(proj))^2)
    addIntr1!(H, 1:length(Latt), - params.h * LocalSpace.Sc)
    initialize!(Map,H,LocalSpace.pspace)
end

# ψ = LGState(Map)
# initialize!(Map,ψ,LocalSpace.pspace)
ψ = LGState(Map,LocalSpace.pspace)

SU!(ψ,H,sualgo)

# H = let H = Hamiltonian()
#     addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
#     addIntr2!(H, ineighbor(Latt;level = 3), LocalSpace.SJ(params.J3 * diagm(ones(3))))
#     initialize!(Map,H,ℂ^2)
# end

# sualgo.τs = [0.01,0.001]
# SU!(ψ,H,sualgo)


O = let obs = Observable()
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
    "E" => measure(ψ,H,sualgo.trunc),
)

@save "Honeycomb spin 1/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@save "Honeycomb spin 1/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

data["E"]
end