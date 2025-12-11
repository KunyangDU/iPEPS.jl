using TensorKit
pspace = ℂ^2
Sx = [0 1;1 0] / 2
Sy = [0 -1im;1im 0] / 2
Sz = [1 0;0 -1] / 2


    
# A = reshape(Sx,2,2,1)

A = TensorMap(Sz,pspace,pspace)
O1_2_O2(A,pspace)
# TensorMap(kron(convert(Array,A),diagm(ones(2))), pspace ⊗ pspace,pspace ⊗ pspace)
