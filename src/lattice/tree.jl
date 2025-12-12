using AbstractTrees

abstract type AbstractTreeNode end
abstract type AbstractObservableTreeNode <: AbstractTreeNode end
# abstract type  <: AbstractTreeNode end

mutable struct LatticeTreeNode <: AbstractTreeNode
    A::Union{Nothing,Tuple}
    parent::Union{Nothing,LatticeTreeNode}
    children::Vector{LatticeTreeNode}

    function LatticeTreeNode(
        A::Union{Nothing,Tuple},
        parent::LatticeTreeNode,
        children::Vector{LatticeTreeNode}=LatticeTreeNode[],
    )
        return new(A,parent,children)
    end

    function LatticeTreeNode(
        A::Union{Nothing,Tuple},
        children::Vector{LatticeTreeNode}=LatticeTreeNode[],
    )
    return new(A,nothing,children)
    end

    # LatticeTreeNode() = LatticeTreeNode()
    # LatticeTreeNode(A::AbstractTensorMap) = LatticeTreeNode(IdentityOperator(A,0))
end

AbstractTrees.nodevalue(node::AbstractTreeNode) = node.A
AbstractTrees.parent(node::AbstractTreeNode) = node.parent
AbstractTrees.children(node::AbstractTreeNode) = node.children
AbstractTrees.ParentLinks(::Type{AbstractTreeNode}) = StoredParents()
AbstractTrees.ChildIndexing(::Type{AbstractTreeNode}) = IndexedChildren()
AbstractTrees.NodeType(::Type{AbstractTreeNode}) = HasNodeType()
AbstractTrees.nodetype(::T) where T <: AbstractTreeNode = T

function addchild!(node::AbstractTreeNode, child::AbstractTreeNode)
    isnothing(child.parent) ? child.parent = node : @assert child.parent == node
    push!(node.children, child)
    return nothing
end

function addchild!(node::T, A::Tuple) where T <: AbstractTreeNode
    addchild!(node,T(A))
    return nothing
end

function cutparent!(node::AbstractTreeNode)
    node.parent = nothing
    return node
end


function Base.show(io::IO, Root::AbstractTreeNode)
    print_tree(Root;maxdepth = 16)
    return nothing
end

isinside(A::Tuple{Int64, Vector{Int64}}, root::LatticeTreeNode) = A  ∈ map(x -> x.A,PreOrderDFS(root))

function findpath(Latt::AbstractLattice, initial::Tuple, target::Tuple, N::Int64 = 4)
    root = LatticeTreeNode(initial)
    buildtree!(root,neighborsites_pbc(Latt),target,N)
    path = Vector[]
    for l in Leaves(root)
        if l.A == target
            tmppath = Tuple[]
            while !isnothing(l)
                push!(tmppath,l.A)
                l = l.parent
            end
            push!(path,tmppath)
        end
    end
    return path
end

function buildtree!(root::LatticeTreeNode, nbsites::Vector, target::Tuple, N::Int64 = 4)
    totalsites = [(1,[0,0]),]

    leaves = collect(Leaves(root))
    while target ∉ map(x -> x.A, leaves)
        childrens = LatticeTreeNode[]
        childrensites = Tuple[]
        for l in leaves
            i,v = l.A
            for (inbs,nbs) in enumerate(map(x -> (x[1],x[2] + v),nbsites[i]))
                nbs in totalsites && continue 
                addchild!(l,nbs)
                push!(childrensites,nbs)
            end
            push!(childrens, l.children...)
        end
        push!(totalsites,childrensites...)
        leaves = childrens
    end
    return root
end

