function generate_bonds(fwd_offsets, basis_idxs=(1,1))
    # basis_idxs: (from, to)
    # 如果 from != to (如 Honeycomb A->B)，则无须生成反向，全向=前向
    if basis_idxs[1] != basis_idxs[2]
        bonds = [Bond(basis_idxs, os) for os in fwd_offsets]
        return bonds, bonds
    else
        # 同一基底，需要生成反向以获得几何上的“所有邻居”
        all_offsets = vcat(fwd_offsets, [-1 .* os for os in fwd_offsets])
        fwd_bonds = [Bond(basis_idxs, os) for os in fwd_offsets]
        all_bonds = [Bond(basis_idxs, os) for os in all_offsets]
        return all_bonds, fwd_bonds
    end
end
# function Base.length(lat::SimpleLattice)
#     N_cells = prod(lat.lattice.L)             # W * L
#     N_basis = length(lat.unitcell.basis_vecs) # 基底数量 (Tri=1, Honey=2)
#     return N_cells * N_basis
# end

reverse_map(sitemap::Tuple) = Tuple([findfirst(x -> x == i,sitemap) for i in Tuple(1:length(sitemap))])

# function group(Latt::AbstractLattice, level0::Vector, level::Int64 = maximum(level0) + 1)
#     nbsites = vcat([neighbor(Latt,1;level = i) for i in level0]...)
#     initialsites = unique(vcat(collect.(nbsites)...))
#     groupsites = Vector[]
#     totalsites = Int64[]
#     to = TimerOutput()
#     for site in initialsites
#         site in totalsites && continue
#         l = 1
#         sublattsites = [site,]
#         while l ≤ length(sublattsites)
#             @timeit to "neighbor" nsites = neighborsites(Latt,sublattsites[l];level = level)
#             @timeit to "filter" push!(sublattsites, filter(x -> x ∉ sublattsites, nsites)...)
#             l += 1
#         end
#         @timeit to "sort" sort!(sublattsites)
#         push!(groupsites, sublattsites)
#         push!(totalsites, sublattsites...)
#     end
#     @assert isequal(1:length(Latt),sort(totalsites))
#     @assert length(groupsites) ≠ 1 "group failed, change the lattice size"
#     show(to)
#     return groupsites
# end

function group(Latt::AbstractLattice, level0::Vector, level::Int64 = maximum(level0) < 3 ? maximum(level0) + 1 : maximum(level0) + 2)
    nbsites = vcat([neighbor(Latt,1;level = i) for i in level0]...)
    initialsites = unique(vcat(collect.(nbsites)...))
    groupsites = Vector[]
    totalsites = Int64[]
    # to = TimerOutput()

    # @timeit to "neighbor" begin 
        nb = neighbor(Latt;level = level)
        nbsites = [Int64[] for _ in 1:length(Latt)]
        for (i,j) in nb 
            push!(nbsites[i],j)
            push!(nbsites[j],i)
        end
        nbsites = Tuple.(Tuple.(nbsites))
    # end
    
    for site in initialsites
        site in totalsites && continue
        l = 1
        sublattsites = [site,]
        while l ≤ length(sublattsites)
            # @timeit to "neighbor" nsites = neighborsites(Latt,sublattsites[l];level = level)
            # @timeit to "filter" 
            push!(sublattsites, filter(x -> x ∉ sublattsites, nbsites[sublattsites[l]])...)
            l += 1
        end
        # @timeit to "sort" 
        sort!(sublattsites)
        push!(groupsites, sublattsites)
        push!(totalsites, sublattsites...)
    end
    @assert isequal(1:length(Latt),sort(totalsites))
    @assert length(groupsites) ≠ 1 "group failed, change the lattice size"
    # show(to)
    return groupsites
end

function FCM(data::Matrix, centers::Matrix, m=2.0)
    n_samples = size(data, 2)
    n_clusters = size(centers, 2)
    weights = zeros(n_clusters, n_samples)
    
    for i in 1:n_samples
        x = data[:, i]
        dists = [norm(x - centers[:, c]) for c in 1:n_clusters]
        
        # FCM 隶属度公式
        for c in 1:n_clusters
            # 避免除以 0
            if dists[c] < 1e-10
                weights[:, i] .= 0.0
                weights[c, i] = 1.0
                break
            end
            
            sum_denom = sum((dists[c] / dists[j])^(2/(m-1)) for j in 1:n_clusters)
            weights[c, i] = 1.0 / sum_denom
        end
    end
    return weights
end

function _magnetic_cell_size(q_frac::Vector{Rational{Int}})
    # 1. 搜集所有满足相位条件的候选向量
    candidates = Vector{Int}[]
    search_range = 5:-1:-5 # 范围通常不需要太大，除非 Q 分母极大
    
    for u1 in search_range, u2 in search_range
        if u1 == 0 && u2 == 0; continue; end
        
        # 检查相位条件: Q . R = 2 * pi * integer
        # 即 h1*u1 + h2*u2 = integer
        phase = q_frac[1] * u1 + q_frac[2] * u2
        if isinteger(phase)
            push!(candidates, [u1, u2])
        end
    end
    
    # 2. 核心优化：先按向量长度排序
    # 这样我们在后面循环时，总是先尝试短向量
    sort!(candidates, by = v -> v[1]^2 + v[2]^2)
    
    min_vol = Inf
    min_len_sum = Inf
    best_vectors = ([], [])
    
    # 3. 寻找线性无关对
    for i in 1:length(candidates)
        for j in (i+1):length(candidates)
            v1 = candidates[i]
            v2 = candidates[j]
            
            # 计算体积 (行列式绝对值)
            vol = abs(v1[1]*v2[2] - v1[2]*v2[1])
            
            # 跳过共线向量 (vol=0)
            if vol == 0; continue; end
            
            # 计算两向量长度平方和，作为次级判据
            len_sum = (v1[1]^2 + v1[2]^2) + (v2[1]^2 + v2[2]^2)
            
            # 更新逻辑：
            # 1. 找到更小的体积 -> 必须更新
            # 2. 体积相同，但找到了更短的向量 -> 也要更新
            if vol < min_vol
                min_vol = vol
                min_len_sum = len_sum
                best_vectors = (v1, v2)
            elseif vol == min_vol && len_sum < min_len_sum
                min_len_sum = len_sum
                best_vectors = (v1, v2)
            end
        end
    end
    
    return Int(min_vol), best_vectors
end

kbasis2(Latt::AbstractLattice) = collect.(eachcol(Latt.unitcell.reciprocal_vecs))
# rbasis2(Latt::AbstractLattice) = collect.(eachcol(vcat(Latt.unitcell.lattice_vecs)))
# rbasis3(Latt::AbstractLattice) = collect.(eachcol(vcat(Latt.unitcell.lattice_vecs,zeros(2)')))
function hoppingvector(Latt::AbstractLattice,i::Int64,j::Int64,v::Vector)
    return coordinate(Latt,j) - coordinate(Latt,i) + Latt.unitcell.lattice_vecs * (size(Latt) .* v)
end

displacement(Latt::AbstractLattice,i::Tuple,j::Tuple) = coordinate(Latt,j...) - coordinate(Latt,j...)
