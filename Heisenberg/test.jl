using TensorKit
include("../src/iPEPS.jl")
# pspace = ℂ^2
# A = TensorMap([0 1;1 0],pspace,pspace)
# B = TensorMap([0 1;1 0],pspace,pspace)

# C = TensorMap(kron([0 1;1 0],[0 1;1 0]),pspace ⊗ pspace,pspace ⊗ pspace)
# C == kron(A,B)
# codom = codomain(C)
# dom = domain(C)

to = TimerOutput()

@timeit to "Hello" rand()
to
# using TensorKit

# # 假设 A 是一个 TensorMap
# # 1. 获取 Codomain (输出/左侧指标) 每个空间的维数
# dims_codomain = dim.(codomain(C))

# # 2. 获取 Domain (输入/右侧指标) 每个空间的维数
# dims_domain = dim.(domain(C))

# # 打印结果
# println("Codomain dims: ", dims_codomain)
# println("Domain dims: ", dims_domain)


# N = 2
# cat(map(x -> reshape(convert(Array,x),dim(codom),1,dim(dom)),[C,C])...;dims = 2)

# codom ⊗ ℂ^2
# dim(cat(map(x -> reshape(convert(Array,x),dim(codom),1,dim(dom)),[C,C])...;dims = 2))
# dim(codom)
# reshape(convert(Array,C),dim(codom),1,dim(dom))
# densify([A,B])
# space(A)[2]
# using TensorKit

# # 1. 定义一个辅助函数生成基矢量 |k> (即 ℂ -> ℂ^d 的 TensorMap)
# function basis_ket(d::Int, k::Int)
#     v = zeros(ComplexF64, ℂ^d, ℂ^1)
#     v[k, 1] = 1.0 
#     return v
# end

# # 2. 构造算符
# # 逻辑：O = ∑ (S_k ⊗ |k>)
# # 注意：S_k ⊗ |k> 的定义域是 pspace ⊗ ℂ，我们需要通过 isomorphism 把它消掉变为 pspace
# S_vec = [Sx, Sy, Sz]
# iso = isomorphism(pspace ⊗ ℂ^1, pspace) # 用于消除定义域中的标量空间 ℂ

# # 这一行代码完成了所有工作，自动处理指标和对称性
# O = sum(S_vec[k] ⊗ basis_ket(3, k) for k in 1:3) * iso'
# 2. 利用张量积直接构造
# 逻辑为：Sx ⊗ <1| + Sy ⊗ <2| + Sz ⊗ <3|
# TensorKit 会自动推导指标：(pspace) ← (pspace) ⊗ (aspace)
# O = Sx ⊗ basis[1]' + Sy ⊗ basis[2]' + Sz ⊗ basis[3]'




