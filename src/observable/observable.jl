mutable struct Observable <: AbstractOperator
    O2::Union{Dict{Tuple,Vector},Dict{Tuple,TensorMap}}
    O1::Union{Dict{Int64,Vector},Dict{Int64,TensorMap}}
    nnnpath::Union{Nothing,Dict{Tuple,Tuple}}
    values::Dict 
    Observable() = new(Dict{Tuple,Vector}(), Dict{Int64,Vector}(),nothing,Dict())
end