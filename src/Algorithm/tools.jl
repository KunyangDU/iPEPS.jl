
function pad(S::AbstractTensorMap, new_space::VectorSpace)
    # 1. 创建一个全零的新张量，定义在更大的空间上
    # S 通常是 (Space) -> (Space)
    S_padded = zeros(eltype(S), new_space, new_space)
    
    # 2. 遍历旧张量的 block，把数据拷过去
    # 注意：只有当 new_space 包含 old_space 的所有 sector 时才有效
    for (c, b) in blocks(S)
        # 获取新张量对应 sector 的 block 引用
        # 注意：这里假设 S_padded 也有 sector c
        # if hasblock(S_padded, c)
        b_padded = block(S_padded, c)
        
        # 拷贝数据 (假设维度是从左上角对齐)
        dims = size(b)
        b_padded[1:dims[1], 1:dims[2]] = b
        # end
    end
    
    return S_padded
end

function diff(A::AbstractTensorMap, B::AbstractTensorMap)
    if rank(A) == rank(B)
        return norm(A - B)
    elseif rank(A) > rank(B)
        return norm(A - pad(B,codomain(A)))
    else
        return norm(B - pad(A,codomain(B)))
    end
end
