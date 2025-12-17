# 定义 Value 的类型别名：任意长度的 Tuple
const TaskPath = Tuple{Vararg{Tuple{Int64, Vector{Int64}}}}



# --- 测试用例 ---

# 构造一个字典，模拟张量网络中的任务
# Key: 任务ID (String 或 Int)
# Value: 涉及的物理索引路径 (模拟 Swap 路径或多体算符)
tasks_dict = Dict(
    "Task_1" => ((1, [0,0]), (2, [0,0]),(3,[1,1])),              # 占用 {1, 2}
    "Task_2" => ((3, [0,0]), (4, [1,0]), (5, [0,0])),  # 占用 {3, 4, 5} -> 可与 Task_1 并行
    "Task_3" => ((2, [0,0]), (3, [1,1])),              # 占用 {2, 3} -> 冲突 Task_1(2), 冲突 Task_2(3)
    "Task_4" => ((6, [0,0]), (1,[0,0])),                         # 占用 {6} -> 单体算符，完全无冲突
    "Task_5" => ((1, [1,1]), (4, [0,1]))               # 占用 {1, 4} -> 冲突 Task_1(1), 冲突 Task_2(4)
)

# 运行分类
batches = partition_dict_tasks(tasks_dict)

# 输出结果
println("并行批次规划结果 (共 $(length(batches)) 批):")
for (i, batch) in enumerate(batches)
    println("Batch $i: $batch")
end