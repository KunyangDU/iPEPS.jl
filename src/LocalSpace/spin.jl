
module TrivialSpinOneHalf

using TensorKit

const pspace = ℂ^2

Sxm = [0 1;1 0] / 2
Sym = [0 -1im;1im 0] / 2
Szm = [1 0;0 -1] / 2

const Sx = TensorMap(Sxm, pspace, pspace)
const Sy = TensorMap(Sym, pspace, pspace)
const Sz = TensorMap(Szm, pspace, pspace)

const SxSx = TensorMap(kron(Sxm,Sxm),pspace ⊗ pspace, pspace ⊗ pspace)
const SySy = TensorMap(kron(Sym,Sym),pspace ⊗ pspace, pspace ⊗ pspace)
const SzSz = TensorMap(kron(Szm,Szm),pspace ⊗ pspace, pspace ⊗ pspace)

const SxSy = TensorMap(kron(Sxm,Sym),pspace ⊗ pspace, pspace ⊗ pspace)
const SySx = TensorMap(kron(Sym,Sxm),pspace ⊗ pspace, pspace ⊗ pspace)

const SxSz = TensorMap(kron(Sxm,Szm),pspace ⊗ pspace, pspace ⊗ pspace)
const SzSx = TensorMap(kron(Szm,Sxm),pspace ⊗ pspace, pspace ⊗ pspace)

const SySz = TensorMap(kron(Sym,Szm),pspace ⊗ pspace, pspace ⊗ pspace)
const SzSy = TensorMap(kron(Szm,Sym),pspace ⊗ pspace, pspace ⊗ pspace)

const SS = SxSx + SySy + SzSz

const Sv = TensorMap(cat(map(x -> reshape(x,2,1,2),(Sxm,Sym,Szm))...;dims = 2), pspace ⊗ ℂ^3, pspace)
const Sh(h::Vector) = h[1] * Sx + h[2] * Sy + h[3] * Sz

const SJ(J::Matrix) = let 
    SSM = [
     SxSx SxSy SxSz;
     SySx SySy SySz;
     SzSx SzSy SzSz
    ]
     return sum(SSM .* J)
end

end

module TrivialSpinOne
using TensorKit
function diagm(A::Pair{Int64, Vector{T}}) where T
    L = abs(A.first) + length(A.second)
    B = zeros(T,L,L)
    for i in 1:length(A.second)
        B[(A.first > 0 ? (i,abs(A.first) + i) : (abs(A.first) + i,i))...] = A.second[i]
    end
    return B
end
diagm(A::Vector) = diagm(0 => A)

const pspace = ℂ^3
const Sz = TensorMap(diagm([1,0,-1]),pspace,pspace)
const S₊ = TensorMap(diagm(1 => sqrt(2)*[1,1]),pspace,pspace)
const S₋ = S₊'
const Sx = (S₊ + S₋) / 2
const Sy = (S₊ - S₋) / 2im 

const SxSx = kron(Sx,Sx)
const SySy = kron(Sy,Sy)
const SzSz = kron(Sz,Sz)
const S2 = TensorMap(diagm(ones(3))*2,pspace,pspace)

const Sx2 = Sx*Sx 
const Sy2 = Sy*Sy 
const Sz2 = Sz*Sz

const Sc = (Sx + Sy + Sz) / sqrt(3)
const Sc2 = Sc*Sc

const SxSy = kron(Sx,Sy)
const SySx = kron(Sy,Sx)
const SySz = kron(Sy,Sz)
const SzSy = kron(Sz,Sy)
const SxSz = kron(Sx,Sz)
const SzSx = kron(Sz,Sx)

const Sh(h::Vector) = h[1] * Sx + h[2] * Sy + h[3] * Sz

const SJ(J::Matrix) = let 
    SSM = [
     SxSx SxSy SxSz;
     SySx SySy SySz;
     SzSx SzSy SzSz
    ]
     return sum(SSM .* J)
end
end


module U₁Spin

using TensorKit

const pspace = Rep[U₁](1//2 => 1, -1//2 => 1)

const Sz = let 
    Op = TensorMap(ones, pspace, pspace )
    block(Op, Irrep[U₁](1//2)) .= 1/2
    block(Op, Irrep[U₁](-1//2)) .= -1/2
    Op
end

const SzSz = kron(Sz, Sz)

const S₊S₋ = let 
    AuxSpace = Rep[U₁](1 => 1)
    OpL = TensorMap(ones, pspace, AuxSpace ⊗ pspace)
    OpR = permute(OpL', ((2,1), (3,)))
    @tensor Op[-1,-2;-3,-4] ≔ OpL[-1,1,-3] * OpR[-2,1,-4]
end

const S₋S₊ = let 
    AuxSpace = Rep[U₁](-1 => 1)
    OpL = TensorMap(ones, pspace, AuxSpace ⊗ pspace)
    OpR = permute(OpL', ((2,1), (3,)))
    @tensor Op[-1,-2;-3,-4] ≔ OpL[-1,1,-3] * OpR[-2,1,-4]
end

const SSxy = (S₊S₋ + S₋S₊) / 2
const SS = SzSz + SSxy

const S2 = TensorMap(ones, Float64, pspace, pspace) * 3 / 4
const Sz2 = Sz*Sz

end
# U₁Spin.SS
# densify([TrivialSpinOneHalf.Sx,TrivialSpinOneHalf.Sy,TrivialSpinOneHalf.Sz]) == TrivialSpinOneHalf.Sv

module SU₂Spin

using TensorKit

const pspace = Rep[SU₂](1//2 => 1)

# S⋅S interaction
const SS = let
    AuxSpace = Rep[SU₂](1 => 1)
    OpL = TensorMap(ones, Float64, pspace, AuxSpace ⊗ pspace) * sqrt(3) / 2.
    OpR = permute(OpL', ((2,1), (3,)))
    @tensor Op[-1,-2;-3,-4] ≔ OpL[-1,1,-3] * OpR[-2,1,-4]
end

const S2 = TensorMap(ones, Float64, pspace, pspace) * 3 / 4

const SS_Casimir = let
    # 两个自旋 1/2 的直积空间    
    # 构造单位算符，稍后我们会根据总自旋 S_tot 修改它的 block
    Op = zeros(Float64, pspace ⊗ pspace, pspace ⊗ pspace)
    
    # 获取 Op 的数据块字典
    # blocks = Op.block/
    
    # 遍历所有融合后的不可约表示 (Irrep)
    # 对于 1/2 x 1/2，Irrep f 将会是 0 (Singlet) 和 1 (Triplet)
    for (f, block) in blocks(Op)
        # f.j 是总自旋 S_tot (例如 0 或 1)
        S_tot = f.j
        
        # S1*S2 = 0.5 * (S_tot(S_tot+1) - S1(S1+1) - S2(S2+1))
        # S(S+1) for spin-1/2 is 3/4
        val = 0.5 * (S_tot * (S_tot + 1) - 0.75 - 0.75)
        
        # 将该 block 的对角元设为计算出的 val
        # block 是一个 Matrix，我们需要加上 val * Identity
        # 这里的 block 维度对应 multiplicity
        n = size(block, 1)
        for i in 1:n
            block[i, i] = val
        end
    end
    Op
end
end


