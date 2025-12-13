
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

# densify([TrivialSpinOneHalf.Sx,TrivialSpinOneHalf.Sy,TrivialSpinOneHalf.Sz]) == TrivialSpinOneHalf.Sv

