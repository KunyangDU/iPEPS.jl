using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 2
Ly = 2
Latt = PeriSqua(Lx,Ly)
@save "Square/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (J1 = 0.0, J2 = 0.1, h = 0.0)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    # addIntr2!(H, ((2,[0,0]),(1,[0,0])), LocalSpace.SJ(params.J1 * diagm(ones(3))))
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
    addIntr2!(H, ineighbor(Latt;level = 2), LocalSpace.SJ(params.J2 * diagm(ones(3))))
    addIntr1!(H,1,LocalSpace.Sh(-[0,0,10000]))
    initialize!(Latt,H,ℂ^2)
end


ψ = LGState(Latt)
initialize!(Latt,ψ,ℂ^2)


D = 3

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    1000,
    [0.1,],
    0.0,
    0.0
)
SU!(ψ,H,sualgo)

# H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
#     addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
#     addIntr2!(H, ineighbor(Latt;level = 2), LocalSpace.SJ(params.J2 * diagm(ones(3))))
#     initialize!(Latt,H,ℂ^2)
# end

# sualgo.τs = [0.01,0.001]
# SU!(ψ,H,sualgo)


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
    "E" => measure(ψ,H,sualgo.trunc),
)

@save "Square/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@save "Square/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

data["E"]
