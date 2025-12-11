O1_2_O2_l(A::AbstractTensorMap,pspace::ElementarySpace) = TensorMap(kron(convert(Array,A),diagm(ones(dim(pspace)))), pspace ⊗ pspace,pspace ⊗ pspace)
O1_2_O2_r(A::AbstractTensorMap,pspace::ElementarySpace) = TensorMap(kron(diagm(ones(dim(pspace))),convert(Array,A)), pspace ⊗ pspace,pspace ⊗ pspace)
