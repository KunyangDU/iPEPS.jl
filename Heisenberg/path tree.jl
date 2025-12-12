using TensorKit
include("../src/iPEPS.jl")

Latt = PeriSqua(2,2)

# coordinate(Latt,2,[-1,1]),coordinate(Latt,1)
# findpath(Latt,(1,[0,0]),(1,[0,1]))
findpath(Latt,(1,[0,0]),(4,[0,0]))
