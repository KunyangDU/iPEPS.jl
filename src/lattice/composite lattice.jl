
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

function ZZPeriHoneycomb(L::Int64, W::Int64, sitemap::Tuple = Tuple(1:4*L*W))
    # --- 1. Define Unit Cell (Horizontal Bonds / Zigzag along Y) ---
    # Bond length a = 1.0
    # Lattice vectors for 4-site rectangular cell
    lx = 3.0
    ly = sqrt(3)
    
    # Basis vectors (4 sites)
    # 1: (0,0) A
    # 2: (1,0) B (Horizontal bond)
    # 3: (1.5, √3/2) A
    # 4: (2.5, √3/2) B
    b1 = [0.0, 0.0]
    b2 = [0.5, sqrt(3)/2]
    b3 = [1.5, sqrt(3)/2]
    b4 = [2.0, 0.0]
    
    unitcell = UnitCell(
        lattice_vecs = [[lx, 0.0], [0.0, ly]],
        basis_vecs   = [b1, b2, b3, b4]
    )
    
    lattice = Lattice(L = [L, W], periodic = [true, true])

    # --- 2. Generate Bonds by Distance Search ---
    # Target distances (squared) for bond length a=1
    # 1NN: d=1 => d^2=1
    # 2NN: d=√3 => d^2=3
    # 3NN: d=2 => d^2=4
    TOL = 1e-4
    d2_targets = Dict(
        1 => 1.0,
        2 => 3.0,
        3 => 4.0
    )
    
    # Storage for bonds
    bonds_all = Dict(1 => Bond[], 2 => Bond[], 3 => Bond[])
    bonds_fwd = Dict(1 => Bond[], 2 => Bond[], 3 => Bond[])
    
    # Search range (enough to cover 3NN)
    search_range = -2:2
    
    for u in 1:4, v in 1:4
        pos_u = unitcell.basis_vecs[u]
        pos_v = unitcell.basis_vecs[v]
        
        for dx in search_range, dy in search_range
            # Offset vector in real space
            # R = dx * a1 + dy * a2
            rx = dx * lx
            ry = dy * ly
            
            # Vector from u to v'
            diff = (pos_v .+ [rx, ry]) .- pos_u
            dist2 = dot(diff, diff)
            
            # Identify Neighbor Order
            order = 0
            for (o, target) in d2_targets
                if abs(dist2 - target) < TOL
                    order = o
                    break
                end
            end
            
            if order > 0
                bond = Bond((u, v), [dx, dy])
                push!(bonds_all[order], bond)
                
                # Filter for Forward Bonds (for non-redundant storage)
                # Logic: keep if u < v, or if u == v and offset is positive
                is_fwd = false
                if u < v
                    is_fwd = true
                elseif u == v
                    # Lexicographic positive check for self-bonds
                    if dx > 0 || (dx == 0 && dy > 0)
                        is_fwd = true
                    end
                end
                
                if is_fwd
                    push!(bonds_fwd[order], bond)
                end
            end
        end
    end

    # --- 3. Construct Bond Dictionary ---
    bond_dict = Dict(
        (true, 1) => bonds_all[1], (false, 1) => bonds_fwd[1],
        (true, 2) => bonds_all[2], (false, 2) => bonds_fwd[2],
        (true, 3) => bonds_all[3], (false, 3) => bonds_fwd[3]
    )

    return CompositeLattice(unitcell, lattice, bond_dict, sitemap)
end

function PeriKagome(L::Int64, W::Int64, sitemap::Tuple = Tuple(1:3*L*W))
    # --- 1. 定义几何参数 (Bond Length a=1) ---
    # Kagome 晶格可以看作是 Triangular 晶格去掉 1/4 节点，
    # 或者看作 3 原子基底的 Triangular 晶格。
    # 为了让最近邻键长为 1.0:
    # 晶格矢量长度应为 2.0
    
    # 晶格矢量 (夹角 60 度)
    lx = 2.0
    ly = sqrt(3) # 对应 2 * sin(60)
    
    a1 = [2.0, 0.0]
    a2 = [1.0, sqrt(3)]
    
    # 基矢 (3 Sites, 构成一个正三角形)
    # 1: (0,0) - 左下
    # 2: (1,0) - 右下
    # 3: (0.5, √3/2) - 顶部
    b1 = [0.0, 0.0]
    b2 = [1.0, 0.0]
    b3 = [0.5, sqrt(3)/2]
    
    unitcell = UnitCell(
        lattice_vecs = [a1, a2],
        basis_vecs   = [b1, b2, b3]
    )
    
    lattice = Lattice(L = [L, W], periodic = [true, true])

    # --- 2. 自动生成 Bond (基于距离搜索) ---
    # 键长 a = 1
    # 1NN: d = 1          => d^2 = 1.0 (三角形边)
    # 2NN: d = √3 ≈ 1.732 => d^2 = 3.0 (跨越六边形)
    # 3NN: d = 2          => d^2 = 4.0 (直线跨越两个三角形)
    
    TOL = 1e-4
    d2_targets = Dict(
        1 => 1.0,
        2 => 3.0,
        3 => 4.0
    )
    
    # 存储容器
    bonds_all = Dict(1 => Bond[], 2 => Bond[], 3 => Bond[])
    bonds_fwd = Dict(1 => Bond[], 2 => Bond[], 3 => Bond[])
    
    # 搜索范围
    search_range = -2:2
    
    for u in 1:3, v in 1:3
        pos_u = unitcell.basis_vecs[u]
        pos_v = unitcell.basis_vecs[v]
        
        for dx in search_range, dy in search_range
            # 计算实空间偏移
            # R = dx * a1 + dy * a2
            rx = dx * a1[1] + dy * a2[1]
            ry = dx * a1[2] + dy * a2[2]
            
            diff = (pos_v .+ [rx, ry]) .- pos_u
            dist2 = dot(diff, diff)
            
            # 判定近邻阶数
            order = 0
            for (o, target) in d2_targets
                if abs(dist2 - target) < TOL
                    order = o
                    break
                end
            end
            
            if order > 0
                bond = Bond((u, v), [dx, dy])
                push!(bonds_all[order], bond)
                
                # 筛选 Forward Bonds
                is_fwd = false
                if u < v
                    is_fwd = true
                elseif u == v
                    # 对于同子格连接 (例如 3NN 直线跳跃)，使用字典序去重
                    if dx > 0 || (dx == 0 && dy > 0)
                        is_fwd = true
                    end
                end
                
                if is_fwd
                    push!(bonds_fwd[order], bond)
                end
            end
        end
    end

    # --- 3. 构建返回对象 ---
    bond_dict = Dict(
        (true, 1) => bonds_all[1], (false, 1) => bonds_fwd[1],
        (true, 2) => bonds_all[2], (false, 2) => bonds_fwd[2],
        (true, 3) => bonds_all[3], (false, 3) => bonds_fwd[3]
    )

    return CompositeLattice(unitcell, lattice, bond_dict, sitemap)
end