O1_2_O2_l(A::AbstractTensorMap,pspace::ElementarySpace) = kron(A,isometry(pspace,pspace))
O1_2_O2_r(A::AbstractTensorMap,pspace::ElementarySpace) = kron(isometry(pspace,pspace),A)

_swap_gate(pspace::ElementarySpace) = permute(id(pspace ⊗ pspace) , (2, 1), (3, 4))

# action(K::AbstractTensorMap{T₁,S₁,3,1},O::AbstractTensorMap{T₂,S₂,2,2},::RIGHT) where {T₁,S₁,T₂,S₂} = @tensor C[-1,-2,-3;-4] ≔ K[-1,1,2,-4] * O[-3,-2,2,1]
action(K::AbstractTensorMap{T₁,S₁,3,1},O::AbstractTensorMap{T₂,S₂,2,2}) where {T₁,S₁,T₂,S₂} = @tensor C[-1,-2,-3;-4] ≔ K[-1,1,2,-4] * O[-3,-2,2,1]
_inner(tmp′::AbstractTensorMap{T,S,3,1},tmp::AbstractTensorMap{T,S,3,1}) where {T,S} = @tensor tmp′[1,3,4,2] * tmp'[2,1,3,4]
function _inner(Γ₁::AbstractTensorMap, O::AbstractTensorMap{T,S,1,1}, Γ₂::AbstractTensorMap) where {T,S}
    return @tensor Γ₁[1,2,3,5,6] * O[4,3] * Γ₂'[5,6,1,2,4]
end

"""
    partition_dict_tasks(task_dict::AbstractDict{K, V})

输入: Dict(Key => Path, ...)
输出: Vector{Vector{K}} (即分好组的 Key 列表)

1. 解析 Dict 中每个任务涉及的物理索引集合。
2. 按“最小物理索引”排序，以优化贪心算法的填充效率。
3. 如果任务路径的索引集合不冲突，则归为同一组。
"""
function partition(task_dict::AbstractDict{K, V}) where {K, V}
    # 1. 预处理：将 Dict 转为包含元数据的数组
    # 我们需要 (Key, 原始Path, 占用索引Set, 最小索引Int)
    annotated_items = Vector{NamedTuple{(:key, :occupied, :min_idx), Tuple{K, Set{Int64}, Int64}}}(undef, length(task_dict))
    
    i = 1
    for (key, path) in task_dict
        # 提取路径中涉及的所有索引
        # path 结构: ((idx1, data), (idx2, data), ...)
        indices = [item[1] for item in path]
        
        annotated_items[i] = (
            key = key,
            occupied = Set(indices),
            # 如果路径为空，设为最大整数放到最后；否则取最小值用于排序
            min_idx = isempty(indices) ? typemax(Int) : minimum(indices)
        )
        i += 1
    end

    # 2. 排序：这是贪心算法得到“最少分组”的关键
    # 优先处理涉及索引较小的任务（通常也是位于整个网络左/上方的任务）
    sort!(annotated_items, by = x -> x.min_idx)

    # 3. 贪心分组
    # 结构: [(Keys列表, 该组占用的总索引Mask), ...]
    groups = Vector{Tuple{Vector{K}, Set{Int64}}}()

    for entry in annotated_items
        current_key = entry.key
        current_indices = entry.occupied
        
        placed = false

        # 尝试放入现有组
        for (group_keys, group_mask) in groups
            # 核心判断：当前任务的点集 与 该组已有点集 是否互斥
            if isempty(intersect(current_indices, group_mask))
                push!(group_keys, current_key)
                union!(group_mask, current_indices) # 更新该组的占用掩码
                placed = true
                break
            end
        end

        # 如果所有现有组都冲突，创建新组
        if !placed
            # 新组初始只包含当前任务
            push!(groups, ([current_key], copy(current_indices)))
        end
    end

    # 4. 提取结果：丢弃 Mask，只返回 Key 的列表
    return [g[1] for g in groups]
end

"""
    partition_disjoint_indices(items::Vector{ComplexInterval})

基于“点集互斥”进行分类。
假设 ((1, ...), (3, ...)) 只占用索引 1 和 3，而不占用 2。
这样 (1, 3) 和 (2, 4) 可以放在同一组，从而大幅减少组数。
"""
function partition(items::Vector)
    # 1. 预处理：提取占用的索引点集，并按最小索引排序
    # 我们将数据转换为 (原始数据, 占用的点集)
    annotated_items = map(items) do item
        # 获取两个端点
        idx1 = item[1][1]
        idx2 = item[2][1]
        # 占用点集：只包含端点
        occupied = Set([idx1, idx2])
        return (item=item, indices=occupied, min_idx=min(idx1, idx2))
    end

    # 排序：优先处理靠前的点，有助于贪心算法找到最优解
    sort!(annotated_items, by = x -> x.min_idx)

    # 存放结果：每组是一个 (Item列表, 该组已占用的总点集)
    # 使用 NamedTuple 或 Tuple 来存储 (items, occupied_mask)
    groups = Vector{Tuple{Vector, Set{Int64}}}()

    for entry in annotated_items
        current_item = entry.item
        current_indices = entry.indices
        
        placed = false

        # 2. 尝试放入现有组
        for (i, (group_items, group_mask)) in enumerate(groups)
            # 检查交集：如果当前点集与该组已占用的点集 无交集
            if isempty(intersect(current_indices, group_mask))
                # 可以放入：添加到列表，并更新该组的占用掩码
                push!(group_items, current_item)
                union!(group_mask, current_indices)
                placed = true
                break
            end
        end

        # 3. 如果放不下，创建新组
        if !placed
            # 新组包含当前项，掩码就是当前项的点集
            new_group_items = [current_item]
            new_group_mask = copy(current_indices)
            push!(groups, (new_group_items, new_group_mask))
        end
    end

    # 返回纯粹的 item 列表 (去掉掩码)
    return [g[1] for g in groups]
end