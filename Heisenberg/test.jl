using TensorKit
include("../src/iPEPS.jl")
pspace = ℂ^2
A = [rand(pspace,pspace) for i in 1:3]
a = A[2]
a0 = deepcopy(a)

a *=2 
A[2] == a0
