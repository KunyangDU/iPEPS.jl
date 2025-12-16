using TensorKit
using LinearAlgebra

# ... (保留你之前的 module SU₂Spin 定义) ...

# === 修正后的验证部分 ===
using .SU₂Spin

# 1. 构造本征态 Projectors (Isometries)
# 结构: [Phys1, Phys2] <-- [Irrep]
# 这是一个 Rank 3 张量: (codom1, codom2) <-- (dom1)
iso_singlet = TensorMap(ones, Float64, SU₂Spin.pspace ⊗ SU₂Spin.pspace, Rep[SU₂](0=>1))
iso_triplet = TensorMap(ones, Float64, SU₂Spin.pspace ⊗ SU₂Spin.pspace, Rep[SU₂](1=>1))

# 归一化
state_singlet = iso_singlet / norm(iso_singlet)
state_triplet = iso_triplet / norm(iso_triplet)

# 2. 计算期望值 <ψ| S⋅S |ψ>
# 注意：state 是 (P, P) <- (I). state' 是 (I) <- (P, P).
# 我们需要把那个 (I) 指标也收缩掉，或者保留它查看对角元。
# 下面代码中，指标 5 就是那个 "Irrep" 通道指标。

# Singlet 期望值
# 这里的 contraction 是: <Singlet| (Index 5) -- [S.S] -- |Singlet> (Index 5)
E_singlet = @tensor state_singlet'[5, 1, 2] * SU₂Spin.SS[1, 2, 3, 4] * state_singlet[3, 4, 5]


# Triplet 期望值
# Triplet 空间是 3维的。上面这种写法会算出 Trace(Energy * Identity) = 3 * Energy。
# 所以我们需要除以维度 dim(Rep(1)) = 3，或者不收缩指标 5 来看对角元。
E_triplet = @tensor state_triplet'[5, 1, 2] * SU₂Spin.SS[1, 2, 3, 4] * state_triplet[3, 4, 5]
# E_triplet_tensor 是一个 1x1 的 scalar tensor (因为 contraction 把所有指标都缩并了)
# 但等一下，Triplet 的 dom 维数是 3。
# 如果我们在 tensor macro 里把两边的 5 连起来，就是在求 Trace。

println("Singlet Energy (Should be -0.75): ", E_singlet)
println("Triplet Energy (Should be +0.25): ", E_triplet)