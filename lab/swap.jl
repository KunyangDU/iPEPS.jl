using TensorKit,Random
include("../src/iPEPS.jl")


Lx = 2
Ly = 2
Latt = PeriSqua(Lx,Ly)
@load "Heisenberg/data/Latt_$(Lx)x$(Ly).jld2" Latt
D = 2
params = (J = 1.0, h = 0.0)
@load "Heisenberg/data/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

paths = (((4, [0, 0]), (2, [0, 0]), (1, [0, 0])), ((4, [0, 0]), (3, [0, 0]), (1, [0, 0])))

# _swap!(ψ,4,3,LEFT(),truncdim(D) & truncbelow(1e-12))
path = ((1, [1, 0]), (3, [1, 0]), (4, [0, 0]))
for path in paths
_swap!(ψ,path[1:end-1],algo.trunc)
_swap!(ψ,reverse(path[1:end-1]),algo.trunc)
end 

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

O.values
# H = let LocalSpace = TrivialSpinOneHalf,H = Hamiltonian()
#     addIntr2!(H, ineighbor(Latt), LocalSpace.SJ(params.J1 * diagm(ones(3))))
#     addIntr2!(H, ineighbor(Latt;level = 2), LocalSpace.SJ(params.J2 * diagm(ones(3))))
#     addIntr1!(H,1,LocalSpace.Sh(-[0,0,100]))
#     initialize!(Latt,H,ℂ^2)
# end

# H.nnnpath[((1, [0, 0]), (4, [0, 0]))]
