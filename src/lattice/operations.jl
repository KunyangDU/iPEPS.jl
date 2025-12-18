function neighbor(Latt::AbstractLattice;level::Int64 = 1,ordered::Bool = false,issort::Bool = true)
    return collect(map(x -> Tuple(issort ? sort(collect(Latt.sitemap[x])) : collect(Latt.sitemap[x])), eachcol(build_neighbor_table(Latt.bond[(ordered,level)], Latt.unitcell, Latt.lattice))))
end
function neighbor(Latt::AbstractLattice,i::Int64;level::Int64 = 1,ordered::Bool = false,issort::Bool = true)
    nb = neighbor(Latt;level = level,ordered = ordered,issort = issort)
    return neighbor(nb,i)
end
neighbor(nb::Vector,i::Int64) = filter(x -> i in x,nb)
neighbor_pbc(nb::Vector,i::Int64) = filter(x -> i in x[1],nb)

function neighborsites(Latt::AbstractLattice,i::Int64;level::Int64 = 1)
    return map(x -> x[1] == i ? x[2] : x[1], neighbor(Latt,i;level = level,ordered = false,issort = true))
end
"""
translation vector on lattice basis
"""
function displacement(Latt::AbstractLattice,i::Int64,j::Int64)
    return collect(sites_to_displacement(i,j,Latt.unitcell,Latt.lattice))
end
"""
translation vector in Descartes coordinate
"""
function distance(Latt::AbstractLattice,i::Int64,j::Int64)
    return collect(displacement_to_vec(displacement(Latt,Latt.asitemap[i],Latt.asitemap[j]),1,1,Latt.unitcell))
end
"""
position on lattice basis with sublattice index
"""
function location(Latt::AbstractLattice,i::Int64)
    return site_to_loc(Latt.asitemap[i],Latt.unitcell,Latt.lattice)
end
"""
position in Descartes coordinate
"""
function coordinate(Latt::AbstractLattice,i::Int64,v::Vector = [0,0])
    return loc_to_pos(Latt[i][2],Latt[i][1],Latt.unitcell) + Latt.unitcell.lattice_vecs * (size(Latt) .* v)
end
# """
#     coordinate(Latt::AbstractLattice, i::Int)

# 获取第 i 个格点的物理坐标 [x, y]。
# 索引顺序假设：基底索引(Basis) -> 宽度方向(W) -> 长度方向(L)
# """
# function coordinate(Latt::AbstractLattice, i::Int)
#     # 1. 获取基本参数
#     # Latt.lattice.L = [W, L]
#     W = Latt.lattice.L[1] 
    
#     a1 = Latt.unitcell.lattice_vecs[1]
#     a2 = Latt.unitcell.lattice_vecs[2]
#     basis_vecs = Latt.unitcell.basis_vecs
    
#     N_basis = length(basis_vecs) # 每个元胞内的格点数 (三角=1, 蜂窝=2)

#     # 2. 索引反解 (Linear Index -> n1, n2, basis_index)
#     # 变成 0-based 索引方便计算
#     idx_0 = i - 1
    
#     # 当前是第几个基底 (1-based)
#     b_idx = (idx_0 % N_basis) + 1
    
#     # 当前是第几个元胞 (0-based)
#     cell_idx = div(idx_0, N_basis)
    
#     # 解析元胞的 grid 坐标 (n1, n2)
#     n1 = cell_idx % W        # 对应 lattice.L[1] 方向
#     n2 = div(cell_idx, W)    # 对应 lattice.L[2] 方向

#     # 3. 计算矢量和
#     # R = n1*a1 + n2*a2 + r_basis
#     coord = n1 .* a1 .+ n2 .* a2 .+ basis_vecs[b_idx]
    
#     return coord
# end
"""
return tb = {L*W*N array}, satisfying tb[i,j,site1] -> site2 which can be obstain by a translation vector [i,j] from site1.
"""
function build_destination_array(Latt::AbstractLattice)
    W,L = Latt.lattice.L
    N = Latt.lattice.N
    tb = zeros(Int64,L,W,N)
    for s in 1:N, i in 1:L,j in 1:W
        tb[i,j,s] = site_to_site(s,[j,i],1,Latt.unitcell,Latt.lattice)
    end
    @assert 0 ∉ tb
    return tb
end
"""
return nbls = {level-th neighbor list}, maps = {array}, satisfying maps[i] = (bond connecting i, sites level-the neighboring i)
"""
function build_neighbor_array(Latt::AbstractLattice;level::Int64 = 1,ordered::Bool = false)
    maps = map_neighbor_table(build_neighbor_table(Latt.bond[(ordered,level)],Latt.unitcell,Latt.lattice))
    maps = [maps[i] for i in 1:Latt.lattice.N]
    nbls = neighbor(Latt;level = level,ordered = ordered)
    return nbls,maps
end
"""
return tb = {L*W*length(states) array}, satisfying tb[i,j,state1] -> state2 which can be obstain by a translation vector [i,j] from state1.
"""
function build_translation_array(Latt::AbstractLattice,states::Vector,intr::Int64=2)
    W,L = Latt.lattice.L
    N = Latt.lattice.N
    dntb = build_destination_array(Latt)
    tb = zeros(Int64,L,W,length(states))
    for i in 1:L, j in 1:W 
        perm = dntb[i,j,:]
        for (is,s) in enumerate(states)
            tb[i,j,is] = statepermute(s,perm,N,intr)
        end
    end
    return tb
end
"""
return tb = {Dict}, satisfying:
- tb[(i,)] -> vector which leads i again, i.e., the smallest periodicity; 
- tb[(i,j)] -> vector which leads j , i.e., the translation vector.
"""
function build_translation_vec_map(Latt::AbstractLattice,states::Vector,intr::Int64=2)
    W,L = Latt.lattice.L
    N = Latt.lattice.N
    dntb = build_destination_array(Latt)
    tb = Dict()
    for i in 1:L, j in 1:W
        perm = dntb[i,j,:]
        for s1 in states
            s2 = statepermute(s1,perm,N,intr)
            if s1 == s2
                exi = get(tb,(s1,),nothing)
                tb[(s1,)] = isnothing(exi) ? [i,j] : (sum(exi) < i+j ? exi : [i,j])
            else
                tb[Tuple(sort([s1,s2]))] = [i,j]
            end
        end
    end
    return tb
end

function _build_neighbor_table(bond::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D}) where {D}

    (; N) = lattice
    (; displacement, orbitals) = bond

    # initialize empty neighbor table
    neighbor_table = []

    # iterate over all unit cells
    for u in 1:N
        # get initial site
        s = loc_to_site(u, orbitals[1], unit_cell)
        # get final site
        s′,pbcv = _site_to_site(s, displacement, orbitals[2], unit_cell, lattice)
        # check if final site was found
        if s′ != 0
            # add to neighbor table
            push!(neighbor_table, ([s,s′],pbcv))
        end
    end

    # if isempty(neighbor_table)
    #     return zeros(Int, 2, 0)
    # else
    #     return hcat(neighbor_table...)
    # end
    return neighbor_table
end


function _site_to_site(s::Int, Δl, o::Int, unit_cell::UnitCell{D}, lattice::Lattice{D}) where {D}

    l = lattice.lvec

    # get unit cell location containing s₁
    o′ = site_to_loc!(l, s, unit_cell, lattice)

    # displace unit cell location
    @. l += Δl
    l₀ = deepcopy(l)

    # apply periodic boundary conditions
    pbc!(l, lattice)

    # get final site
    s′ = loc_to_site(l, o, unit_cell, lattice)

    return s′,collect(l₀ - l)
end

function _build_neighbor_table(bonds, unit_cell::UnitCell{D}, lattice::Lattice{D}) where {D}

    neighbor_tables = []
    for i in eachindex(bonds)
        neighbor_table = _build_neighbor_table(bonds[i], unit_cell, lattice)
        push!(neighbor_tables, neighbor_table...)
    end

    return neighbor_tables
end
"""
((i,j),v), v: i -> j lattice vector, real R = coord(j-i) + basis*v
"""
function neighbor_pbc(Latt::AbstractLattice;level::Int64 = 1,ordered::Bool = false, issort::Bool = false)
    nb = _build_neighbor_table(Latt.bond[(ordered,level)],Latt.unitcell,Latt.lattice)
    if issort
        nb =  map(x -> (Tuple(sort(collect(x[1]))),x[2]), nb)
    end
    return map(x -> (Tuple(Latt.sitemap[x[1]]),x[2]), nb)
end

function neighbor_pbc(Latt::AbstractLattice,i::Int64;level::Int64 = 1,ordered::Bool = false,issort::Bool = false)
    nb = neighbor_pbc(Latt;level = level,ordered = ordered,issort = issort)
    return neighbor_pbc(nb,i)
end

# function neighborsites_pbc(Latt::AbstractLattice,i::Int64;level::Int64 = 1)
#     return map(x -> (x[1][1] == i ? x[1][2] : x[1][1],x[2]), neighbor_pbc(Latt,i;level = level,ordered = false,issort = false))
# end
function neighborsites_pbc(Latt::AbstractLattice;level::Int64 = 1,banlist::Vector = [])
    nbsites = [[] for i in 1:length(Latt)]
    nbpairs = neighbor_pbc(Latt;level = level, issort = false, ordered = false)
    for ((i,j),v) in nbpairs
        v = collect(Int.( v ./ size(Latt)))
        ((i,[0,0]),(j,v)) ∈ banlist && continue
        ((j,[0,0]),(i,-v)) ∈ banlist && continue
        push!(nbsites[i], (j,v))
        push!(nbsites[j], (i,-v))
    end
    return nbsites
end

function ineighbor(Latt::AbstractLattice;level::Int64 = 1,ordered::Bool = false)
    nbs = Tuple[]
    for ((i,j),v) in neighbor_pbc(Latt;level = level, issort = false, ordered = ordered)
        v = collect(Int.( v ./ size(Latt)))
        push!(nbs,((i,[0,0]), (j,v)))
    end
    return nbs
end
