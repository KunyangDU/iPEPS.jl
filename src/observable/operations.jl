
# function densify(A::Vector,codom::ProductSpace = codomain(A[1]), dom::ProductSpace = codomain(A[1]))
#     N = length(A)
#     return TensorMap(cat(map(x -> reshape(convert(Array,x),dim.(codom)...,1,dim.(dom)...),A)...;dims = length(codom) + 1), codom ⊗ ℂ^N, dom)
#     # return TensorMap(cat(map(x -> reshape(x,dim(codom),1,dim(dom)),convert.(Array,A))...;dims = 2), codom ⊗ ℂ^N, dom)
# end

# function densify!(O::Observable)
#     vsd = (O.O1)
#     O.O1 = Dict{Int64,TensorMap}()
#     for (i,os) in vsd
#         O.O1[i] = densify(os)
#     end

#     vsd = O.O2
#     O.O2 = Dict{Tuple,TensorMap}()
#     for (nb,os) in vsd
#         O.O2[nb] = densify(os)
#     end
#     return O
# end