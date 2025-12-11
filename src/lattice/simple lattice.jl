
mutable struct SimpleLattice{D,S,L} <: AbstractLattice
    unitcell::LatticeUtilities.UnitCell
    lattice::LatticeUtilities.Lattice
    bond::Dict
    sitemap::Tuple
    asitemap::Tuple
    group::Union{Nothing,Tuple}
    function SimpleLattice(unitcell::UnitCell,lattice::Lattice{D},bond::Dict,sitemap::Tuple = Tuple(1:*(lattice.L...))) where D
        @assert unitcell.n == 1
        S = lattice.L
        asitemap = reverse_map(sitemap)
        return new{D,S,*(S...)}(unitcell,lattice,bond,sitemap,asitemap,nothing)
    end
end

Base.length(::SimpleLattice{D,S,L}) where {D,S,L} = L

function PeriSqua(L::Int64, W::Int64, sitemap::Tuple = Tuple(1:L*W))
    # 定义正方晶格元胞
    sq = UnitCell(lattice_vecs = [[1., 0.], [0., 1.]], basis_vecs = [[0., 0.]])
    lattice = Lattice(L = [L, W], periodic = [true, true])

    # --- 定义前向偏移量 (Forward Offsets) ---
    
    # 1NN (d^2 = 1): (1,0), (0,1)
    os_1 = [[1, 0], [0, 1]]
    
    # 2NN (d^2 = 2): (1,1), (1,-1) [对角线]
    os_2 = [[1, 1], [1, -1]]
    
    # 3NN (d^2 = 4): (2,0), (0,2) [沿轴跳一步]
    os_3 = [[2, 0], [0, 2]]
    
    # 4NN (d^2 = 5): (2,1), (1,2), (2,-1), (1,-2) [马步]
    os_4 = [[2, 1], [1, 2], [2, -1], [1, -2]]
    
    # 5NN (d^2 = 8): (2,2), (2,-2) [对角跳一步]
    os_5 = [[2, 2], [2, -2]]
    
    # 6NN (d^2 = 9): (3,0), (0,3) [沿轴跳两步]
    os_6 = [[3, 0], [0, 3]]

    # --- 生成 Bond ---
    b1_all, b1_fwd = generate_bonds(os_1)
    b2_all, b2_fwd = generate_bonds(os_2)
    b3_all, b3_fwd = generate_bonds(os_3)
    b4_all, b4_fwd = generate_bonds(os_4)
    b5_all, b5_fwd = generate_bonds(os_5)
    b6_all, b6_fwd = generate_bonds(os_6)

    bond = Dict(
        # Order 1
        (true, 1) => b1_all, (false, 1) => b1_fwd,
        # Order 2
        (true, 2) => b2_all, (false, 2) => b2_fwd,
        # Order 3
        (true, 3) => b3_all, (false, 3) => b3_fwd,
        # Order 4
        (true, 4) => b4_all, (false, 4) => b4_fwd,
        # Order 5
        (true, 5) => b5_all, (false, 5) => b5_fwd,
        # Order 6
        (true, 6) => b6_all, (false, 6) => b6_fwd,
    )

    return SimpleLattice(sq, lattice, bond, sitemap)
end

function XCPeriTria(L::Int64, W::Int64, sitemap::Tuple = Tuple(1:L*W))
    tria = UnitCell(lattice_vecs = [[1., 0.], [0.5, sqrt(3)/2]], basis_vecs = [[0., 0.]])
    lattice = Lattice(L = [L, W], periodic = [true, true])

# --- 偏移量定义 ---
    
    # Order 1 (d^2=1): 6 neighbors
    os_1 = [[1, 0], [0, 1], [-1, 1]]
    
    # Order 2 (d^2=3): 6 neighbors
    os_2 = [[1, 1], [1, -2], [2, -1]]
    
    # Order 3 (d^2=4): 6 neighbors (2 * os_1)
    os_3 = [[2, 0], [0, 2], [-2, 2]]
    
    # Order 4 (d^2=7): 12 neighbors -> 6 forward offsets
    # 类似正方晶格的"马步"，但在三角晶格中方向更多
    os_4 = [[2, 1], [1, 2], [3, -1], [3, -2], [1, -3], [2, -3]]
    
    # Order 5 (d^2=9): 6 neighbors (3 * os_1)
    os_5 = [[3, 0], [0, 3], [-3, 3]]
    
    # Order 6 (d^2=12): 6 neighbors (2 * os_2)
    os_6 = [[2, 2], [2, -4], [4, -2]]

    # --- 生成 Bonds ---
    b1_all, b1_fwd = generate_bonds(os_1)
    b2_all, b2_fwd = generate_bonds(os_2)
    b3_all, b3_fwd = generate_bonds(os_3)
    b4_all, b4_fwd = generate_bonds(os_4)
    b5_all, b5_fwd = generate_bonds(os_5)
    b6_all, b6_fwd = generate_bonds(os_6)

    bond = Dict(
        (true, 1) => b1_all, (false, 1) => b1_fwd,
        (true, 2) => b2_all, (false, 2) => b2_fwd,
        (true, 3) => b3_all, (false, 3) => b3_fwd,
        (true, 4) => b4_all, (false, 4) => b4_fwd,
        (true, 5) => b5_all, (false, 5) => b5_fwd,
        (true, 6) => b6_all, (false, 6) => b6_fwd,
    )

    return SimpleLattice(tria, lattice, bond, sitemap)
end
