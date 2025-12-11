
mutable struct CompositeLattice{D,S,L} <: AbstractLattice
    unitcell::LatticeUtilities.UnitCell
    lattice::LatticeUtilities.Lattice
    bond::Dict
    sitemap::Tuple
    asitemap::Tuple
    group::Union{Nothing,Tuple}
    function CompositeLattice(unitcell::UnitCell,lattice::Lattice{D},bond::Dict,sitemap::Tuple = Tuple(1:*(lattice.L...) * unitcell.n)) where D
        asitemap = Tuple([findfirst(x -> x == i,sitemap) for i in Tuple(1:*(lattice.L...) * unitcell.n)])
        return new{D,lattice.L,*(lattice.L...) * unitcell.n}(unitcell,lattice,bond,sitemap,asitemap,nothing)
    end
end

Base.length(::CompositeLattice{D,S,L}) where {D,S,L} = L

function XCPeriHoneycomb(L::Int64, W::Int64, sitemap::Tuple = Tuple(1:2L*W))
honey = UnitCell(
        lattice_vecs = [[1., 0.], [0.5, sqrt(3)/2]],
        basis_vecs   = [[0., 0.], [0.5, sqrt(3)/6]] 
    )
    lattice = Lattice(L = [L, W], periodic = [true, true])

    # --- 辅助变量 ---
    # 三角晶格前向 (用于同子格 A-A 连接)
    tri_fwd_1 = [[1, 0], [0, 1], [-1, 1]]           # d^2=1
    tri_fwd_2 = [[1, 1], [1, -2], [2, -1]]          # d^2=3

    # --- 偏移量定义 ---

    # Order 1 (A->B, d^2=1/3): 3 neighbors
    os_1 = [[0, 0], [-1, 0], [0, -1]]

    # Order 2 (A->A / B->B, d^2=1): 6 neighbors
    # 使用 tri_fwd_1

    # Order 3 (A->B, d^2=4/3): 3 neighbors
    os_3 = [[1, -1], [-1, 1], [-1, -1]]

    # Order 4 (A->B, d^2=7/3): 6 neighbors
    os_4 = [[1, 0], [0, 1], [-2, 1], [-1, 2], [1, -2], [2, -1]]

    # Order 5 (A->A / B->B, d^2=3): 6 neighbors
    # 使用 tri_fwd_2

    # Order 6 (A->B, d^2=13/3): 3 neighbors
    # 几何上对应 3NN 的更远延伸
    os_6 = [[2, -2], [-2, 2], [-2, -2]]

    # --- 生成 Bonds ---
    
    # A->B 类型 (Order 1, 3, 4, 6)
    b1_all, _ = generate_bonds(os_1, (1, 2))
    b3_all, _ = generate_bonds(os_3, (1, 2))
    b4_all, _ = generate_bonds(os_4, (1, 2))
    b6_all, _ = generate_bonds(os_6, (1, 2))

    # A->A / B->B 类型 (Order 2, 5)
    # 2NN
    b2a_all, b2a_fwd = generate_bonds(tri_fwd_1, (1, 1))
    b2b_all, b2b_fwd = generate_bonds(tri_fwd_1, (2, 2))
    b2_all = vcat(b2a_all, b2b_all)
    b2_fwd = vcat(b2a_fwd, b2b_fwd)
    
    # 5NN
    b5a_all, b5a_fwd = generate_bonds(tri_fwd_2, (1, 1))
    b5b_all, b5b_fwd = generate_bonds(tri_fwd_2, (2, 2))
    b5_all = vcat(b5a_all, b5b_all)
    b5_fwd = vcat(b5a_fwd, b5b_fwd)

    bond = Dict(
        # Order 1 (AB)
        (true, 1) => b1_all, (false, 1) => b1_all,
        # Order 2 (AA/BB)
        (true, 2) => b2_all, (false, 2) => b2_fwd,
        # Order 3 (AB)
        (true, 3) => b3_all, (false, 3) => b3_all,
        # Order 4 (AB)
        (true, 4) => b4_all, (false, 4) => b4_all,
        # Order 5 (AA/BB)
        (true, 5) => b5_all, (false, 5) => b5_fwd,
        # Order 6 (AB)
        (true, 6) => b6_all, (false, 6) => b6_all,
    )
    
    return CompositeLattice(honey, lattice, bond, sitemap)
end