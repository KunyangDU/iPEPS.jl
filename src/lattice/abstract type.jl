abstract type AbstractLattice end
# abstract type SimpleLattice{D,S,L} <: AbstractLattice end

Base.@propagate_inbounds Base.getindex(A::AbstractLattice, i::Int) = reverse(location(A,i))
Base.size(A::AbstractLattice) = A.lattice.L
TensorKit.dim(A::AbstractLattice) = dim(A.lattice)
TensorKit.dim(::Lattice{D}) where D = D
# Base.@propagate_inbounds Base.setindex!(A::SimpleLattice, value, i::Int) = 1



