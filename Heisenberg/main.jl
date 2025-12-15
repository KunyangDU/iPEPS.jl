using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 4
Ly = 4
Latt = PeriSqua(Lx,Ly)
@save "Heisenberg/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (J = 1.0, h = 0.0)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J * diagm(ones(3))))
    addIntr1!(H,1,LocalSpace.Sh(-[0,0,100]))
    initialize!(Latt,H,ℂ^2)
end


ψ = LGState(Latt)
initialize!(Latt,ψ,ℂ^2)


D = 2

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    3000,
    [0.1,],
    0.0,
    0.0
)
SU!(ψ,H,sualgo)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J * diagm(ones(3))))
    initialize!(Latt,H,ℂ^2)
end

sualgo.τs = [0.01,0.001]
SU!(ψ,H,sualgo)


O = let obs = Observable(), LocalSpace = TrivialSpinOneHalf
    addObs1!(obs,1:length(Latt),LocalSpace.Sx)
    addObs1!(obs,1:length(Latt),LocalSpace.Sy)
    addObs1!(obs,1:length(Latt),LocalSpace.Sz)
    addObs2!(obs,ineighbor(Latt),LocalSpace.SxSx)
    addObs2!(obs,ineighbor(Latt),LocalSpace.SySy)
    addObs2!(obs,ineighbor(Latt),LocalSpace.SzSz)
    initialize!(Latt,obs)
end

calObs!(O,ψ)
data = Dict(
    "Obs" => O.values,
    "E" => measure(ψ,H),
)

@save "Heisenberg/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

data["E"]