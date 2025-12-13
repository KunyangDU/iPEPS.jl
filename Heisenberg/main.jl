using TensorKit
include("../src/iPEPS.jl")

Lx = 2
Ly = 2
Latt = PeriSqua(Lx,Ly)
@save "Heisenberg/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2

ψ = let pspace = TrivialSpinOneHalf.pspace, aspace = ℂ^1
    Dict(
        "Γ" => [rand(ComplexF64, aspace ⊗ aspace ⊗ pspace, aspace ⊗ aspace) for i in 1:Lx, j in 1:Ly],
        "λu" => [(isometry(ComplexF64, aspace, aspace)) for i in 1:Lx, j in 1:Ly],
        "λr" => [(isometry(ComplexF64, aspace, aspace)) for i in 1:Lx, j in 1:Ly]
    )
end


params = (J = 1.0,)
H = Dict(
    "sites1" => [1,],
    "sites2" => neighbor_pbc(Latt;issort = false),
    "H2" => params.J * TrivialSpinOneHalf.SS,
)
H["sites1nb"] = map(x -> length(filter(y -> x in y[1],H["sites2"])), H["sites1"])

SUalgo = Dict(
    "ϵ" => 1e-8,
    "D" => D,
    "tol" => 1e-4,
    "N" => 3000,
    "τs" => [0.01,0.001],
    "τ" => 0.0,
)

# params = (J = 1.0, h = 100.0)
# H["H1"] = params.h * TrivialSpinOneHalf.Sh(-[0.0,0.0,1.0])
# SUalgo["τs"] = [0.1,]
# SUupdate!(ψ,H,Latt,SUalgo)

params = (J = 1.0, h = 0.0)
H["H1"] = params.h * TrivialSpinOneHalf.Sh(-[0.0,0.0,1.0])
SUalgo["τs"] = [0.01,0.001]
SUupdate!(ψ,H,Latt,SUalgo)

Obs = Dict(
    "S" => [],
    "E" => measure2(ψ,H) / length(Latt)
)
for i in 1:length(Latt)
    O = TrivialSpinOneHalf.Sv
    λr, λu, λd, λl= λs(ψ,Latt,i)
    Γ = ψ["Γ"][(Latt[i][2] + [1,1])...]
    Γ′ = λΓcontract(Γ, λr, λu, λd, λl)
    tmp = _inner(Γ′,O,Γ′)
    push!(Obs["S"], real(convert(Array,tmp)))
end


@save "Heisenberg/data/Obs_$(Lx)x$(Ly)_$(D)_$(params).jld2" Obs

@show Obs["E"]
Obs["S"]
λs(ψ,Latt,1)