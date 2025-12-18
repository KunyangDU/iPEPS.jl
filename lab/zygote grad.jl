using TensorKit, Zygote, LinearAlgebra
include("../src/iPEPS.jl")
# const SS = let pspace = Rep[SU₂](1/2 => 1)
#     AuxSpace = Rep[SU₂](1 => 1)
#     OpL = TensorMap(ones, Float64, pspace, AuxSpace ⊗ pspace) * sqrt(3) / 2.
#     OpR = permute(OpL', ((2,1), (3,)))
#     @tensor Op[-1,-2;-3,-4] ≔ OpL[-1,1,-3] * OpR[-2,1,-4]
# end

# 定义你原本的 Loss
function loss_fn_original(T)
    # 注意：tsvd 返回 (U,S,V)，我们需要取 [2] 才是 S
    # 如果你直接写 S=tsvd(...)，那 norm(S) 实际上是在求 (U,S,V) 三个张量整体的范数
    # 但结果也是趋向于 0
    # return real(@tensor T[1,2,3] * T'[3,1,2])
    # vals = tsvd(T, (1,2), (3,)) 
    # S = vals[2] 
    # # return real(@tensor S[1,2] * S'[2,1])
    # return norm(S) ^ 2
    H = TrivialSpinOneHalf.SS
    T′ = T / norm(T)
    return real(@tensor T′[1,2] * H[3,4,1,2] * T′'[3,4])
end

A = normalize(rand(ComplexF64, ℂ^2 ⊗ ℂ^2))
η = 0.1

println("Initial Loss: $(loss_fn_original(A))")

for i in 1:100
    val, grads = withgradient(loss_fn_original, A)
    global A -= η * grads[1]
    if i % 10 == 0
        println("Step $i: Loss = $val")
    end
end

println("Final Loss: $(loss_fn_original(A)) (Should be close to 0)")