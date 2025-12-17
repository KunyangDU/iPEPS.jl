# nnnpath: physics Lattice
mutable struct Hamiltonian <: AbstractOperator
    H2::Dict{Tuple,TensorMap}
    H1::Dict{Int64,TensorMap}
    nnnpath::Union{Nothing,Dict{Tuple,Tuple}}
    coordination::Tuple
    partition::Vector
    pspace::ElementarySpace
    function Hamiltonian(H2::Dict{Tuple,TensorMap},
    H1::Dict{Int64,TensorMap})
        return new(H2,H1,nothing,(),[],ℂ^1)
    end
    Hamiltonian() = new(Dict{Tuple,TensorMap}(),Dict{Int64,TensorMap}(),nothing,(),[],ℂ^1)
end


