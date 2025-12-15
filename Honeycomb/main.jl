using TensorKit,Random
include("../src/iPEPS.jl")

Lx = 2
Ly = 2
Latt = ZZPeriHoneycomb(Lx,Ly)
@save "Honeycomb/data/Latt_$(Lx)x$(Ly).jld2" Latt
auxLatt = PeriSqua(2Lx,2Ly,ZZHCmap(Latt))

params = (J1 = -1.0, J3 = 0.3, h = 0.0)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
    addIntr2!(H, ineighbor(Latt;level = 3), LocalSpace.SJ(params.J3 * diagm(ones(3))))
    addIntr1!(H,1,LocalSpace.Sh(-[0,0,100]))
    initialize!(Latt,H,ℂ^2,_fullize(ineighbor(auxLatt)))
end

# A = ineighbor(Latt) |> x -> vcat(x,_nn_reverse.(x))
# B = ineighbor(auxLatt) |> x -> vcat(x,_nn_reverse.(x))
# setdiff(A,B)

ψ = LGState(auxLatt)
initialize!(auxLatt,ψ,ℂ^2)

D = 2

sualgo = SimpleUpdate(
    truncdim(D) & truncbelow(1e-12),
    1e-4,
    1000,
    [0.1,],
    0.0,
    0.0
)

SU!(ψ,H,sualgo)

H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
    addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
    addIntr2!(H, ineighbor(Latt;level = 3), LocalSpace.SJ(params.J3 * diagm(ones(3))))
    initialize!(Latt,H,ℂ^2,_fullize(ineighbor(auxLatt)))
end

sualgo.τs = [0.1,0.01,0.001]
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

@save "Honeycomb/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@save "Honeycomb/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

data["E"]
