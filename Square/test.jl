using TensorKit

using TensorKit
using LinearAlgebra

using TensorKit
using LinearAlgebra

"""
    svd_dist_sq(S1::AbstractTensorMap, S2::AbstractTensorMap)

计算两个张量（支持对角张量 DiagonalTensorMap）之间的“距离”平方。
兼容 Symmetric (如 SU2) 和 Non-Symmetric (如 ComplexSpace) 情况。
"""
function Base.diff(S1::AbstractTensorMap, S2::AbstractTensorMap)
    dist_sq = 0.0
    
    # 1. 获取两个张量包含的所有量子数 (Sectors)
    # 使用 Set 进行快速查找 (O(1) lookup)
    secs1 = Set(blocksectors(S1))
    secs2 = Set(blocksectors(S2))
    
    # 取并集
    all_sectors = union(secs1, secs2)
    
    for c in all_sectors
        # 2. 检查 Sector 是否存在于各自的 Set 中
        b1_exists = c in secs1
        b2_exists = c in secs2
        
        if b1_exists && b2_exists
            b1 = block(S1, c)
            b2 = block(S2, c)
            dist_sq += _calc_block_dist(b1, b2)
            
        elseif b1_exists
            dist_sq += norm(block(S1, c))^2
        elseif b2_exists
            dist_sq += norm(block(S2, c))^2
        end
    end
    
    return dist_sq
end

# 辅助函数：计算两个 Block 之间的带 Padding 距离
function _calc_block_dist(b1::AbstractArray, b2::AbstractArray)
    # Case A: 两者都是 Vector (DiagonalTensorMap 的典型情况 - 奇异值)
    if b1 isa AbstractVector && b2 isa AbstractVector
        n1, n2 = length(b1), length(b2)
        n_min = min(n1, n2)
        
        # 两个向量在重叠部分的内积
        overlap = dot(view(b1, 1:n_min), view(b2, 1:n_min))
        
        return norm(b1)^2 + norm(b2)^2 - 2 * real(overlap)

    # Case B: 两者都是 Matrix (普通 TensorMap)
    elseif b1 isa AbstractMatrix && b2 isa AbstractMatrix
        r1, c1 = size(b1)
        r2, c2 = size(b2)
        r_min, c_min = min(r1, r2), min(c1, c2)
        
        # 两个矩阵在左上角重叠部分的内积
        overlap = dot(view(b1, 1:r_min, 1:c_min), view(b2, 1:r_min, 1:c_min))
        
        return norm(b1)^2 + norm(b2)^2 - 2 * real(overlap)
        
    else
        # Case C: 混合情况 (例如对角阵 vs 稠密阵)
        # 提取对角元进行计算
        v1 = b1 isa AbstractMatrix ? diag(b1) : b1
        v2 = b2 isa AbstractMatrix ? diag(b2) : b2
        return _calc_block_dist(v1, v2)
    end
end



A = tsvd(rand(ℂ^2⊗ℂ^2,ℂ^3))[2]
B = tsvd(rand(ℂ^2⊗ℂ^2,ℂ^2))[2]
Asu2 = tsvd(rand(Rep[SU₂](0 => 2, 1=> 2), Rep[SU₂](1/2 => 2, 3/2=> 2) ⊗ Rep[SU₂](1/2 => 2)))[2]
Bsu2 = tsvd(rand(Rep[SU₂](1/2 => 2, 3/2=> 2), Rep[SU₂](0 => 2, 1 => 2, 2=> 2) ⊗ Rep[SU₂](1/2 => 2)))[2]
Au1 = tsvd(rand(Rep[U₁](0 => 2, 1=> 2), Rep[U₁](1/2 => 2, 3/2=> 2) ⊗ Rep[U₁](1/2 => 2)))[2]
Bu1 = tsvd(rand(Rep[U₁](1/2 => 2, 3/2=> 2), Rep[U₁](0 => 2, 1 => 2, 2=> 2) ⊗ Rep[U₁](1/2 => 2)))[2]

diff(A,B)
diff(Asu2, Bsu2)
diff(Au1, Bu1)
