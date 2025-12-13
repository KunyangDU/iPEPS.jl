using TensorKit
include("../src/iPEPS.jl")
pspace = ℂ^2
A = TensorMap([0 1;1 0],pspace,pspace)
B = TensorMap([0 1;1 0],pspace,pspace)

C = TensorMap(kron([0 1;1 0],[0 1;1 0]),pspace ⊗ pspace,pspace ⊗ pspace)
C == kron(A,B)



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




