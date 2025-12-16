using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 3
Ly = 3
Latt = XCPeriTria(Lx,Ly)
@save "Triangular/data/Latt_$(Lx)x$(Ly).jld2" Latt
Map = XCPeriTriaMapping(Latt)
# for h in 0.2:0.4:5.0
params = (Jxy = 1.0, Jz = 1.68, h = 0.0)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(diagm([params.Jxy,params.Jxy,params.Jz])))
    addIntr1!(H,1:length(Latt),params.h * LocalSpace.Sh(-[0,0,1]))
    addIntr1!(H,1,100 * LocalSpace.Sh(-[0,0,1]))
    initialize!(Map,H,ℂ^2)
end

ψ = LGState(Map)
initialize!(Map,ψ,ℂ^2)

D = 3

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    2000,
    [0.1,],
    0.0,
    0.0
)


SU!(ψ,H,sualgo)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(diagm([params.Jxy,params.Jxy,params.Jz])))
    addIntr1!(H,1:length(Latt),params.h * LocalSpace.Sh(-[0,0,1]))
    initialize!(Map,H,ℂ^2)
end

sualgo.τs = [0.01,0.001]
SU!(ψ,H,sualgo)


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
    "E" => measure(ψ,H,sualgo.trunc),
)

@save "Triangular/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@save "Triangular/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

data["E"]
# end
