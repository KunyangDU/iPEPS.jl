using LatticeUtilities
using JLD2,TimerOutputs
using CairoMakie,LaTeXStrings,ColorSchemes


include("tools.jl")
include("lattice/abstract type.jl")
include("ProcessControl/type.jl")

include("Algorithm/SU.jl")
include("Algorithm/contract.jl")
include("Algorithm/tools.jl")

include("LocalSpace/spin.jl")
include("Observables/calObs.jl")
include("Operators/operations.jl")
include("Operators/hamiltonian.jl")
include("State/operations.jl")

include("lattice/simple lattice.jl")
include("lattice/composite lattice.jl")
include("lattice/tools.jl")
include("lattice/operations.jl")
include("lattice/plot.jl")
include("lattice/geometry.jl")


