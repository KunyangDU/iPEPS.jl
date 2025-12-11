
function plotLatt!(ax::Axis,Latt::AbstractLattice,
    # xshift::Vector = collect.(eachcol(Latt.unitcell.lattice_vecs))[1],
    # yshift::Vector = collect.(eachcol(Latt.unitcell.lattice_vecs))[2],
    ;kwargs...)

    Lx,Ly = size(Latt)

    bond = get(kwargs, :bond, true)
    tplevel = get(kwargs, :tplevel, (1))
    site = get(kwargs, :site, false)
    selectedsite = get(kwargs,:selectedsite,1:length(Latt))
    sitelabel = get(kwargs, :sitelabel, true)
    sitesize = get(kwargs, :sitesize, 16 .* ones(length(selectedsite)))
    sitecolor = get(kwargs, :sitecolor, [:grey for _ in 1:length(selectedsite)])
    sitealpha = get(kwargs, :sitealpha, ones(length(selectedsite)))
    total_shift = get(kwargs, :total_shift, [0,0])
    
    linewidth = get(kwargs, :linewidth, 2)
    linecolor = get(kwargs, :linecolor, RGBf(0.5, 0.5, 0.5))


    if bond
        for level in tplevel
            # NN bond 
            for ((i, j),v) in get(kwargs,:pairs,neighbor_pbc(Latt;level = level,issort = false))

                    x = map([i, j]) do i
                        coordinate(Latt, i)[1] + total_shift[1]
                    end
                    y = map([i, j]) do i
                        coordinate(Latt, i)[2] + total_shift[2]
                    end

                    x[2] += (Latt.unitcell.lattice_vecs * v)[1]
                    y[2] += (Latt.unitcell.lattice_vecs * v)[2]
                    # x[1] -= v[1] * xshift[1] + v[1] * yshift[1]
                    # y[1] -= v[1] * xshift[2] + v[2] * yshift[2]

                    # if abs((coordinate(Latt,i) .- coordinate(Latt,j))[1]) > (Lx-1)*xshift[1] - 1e-5 
                    #     x[findmin(x)[2]] += xshift[1] * Lx
                    #     y[findmin(y)[2]] += xshift[2] * Lx
                    # end

                    # if abs((coordinate(Latt,i) .- coordinate(Latt,j))[2]) > (Ly-1)*yshift[2] - 1e-5 
                    #     x[findmin(x)[2]] += yshift[1] * Ly
                    #     y[findmin(y)[2]] += yshift[2] * Ly
                    # end

                    lines!(ax, x, y;
                        linewidth=linewidth,
                        color=linecolor,
                    )
            end
        end
    end

    if site
        for (i,s) in enumerate(selectedsite)
                x, y = coordinate(Latt, s)

                CairoMakie.scatter!(ax, x + total_shift[1], y + total_shift[2];
                    markersize=sitesize[i],
                    color=sitecolor[i],
                    alpha = sitealpha[i])
        
                sitelabel && text!(ax, x + 0.05 + total_shift[1], y + 0.05 + total_shift[2]; text = "$s")
        end
    end
    
end


function arrow0!(ax::Axis,x, y, u, v; arrowsize=0.386, color=:black, transparency=1,linewidth = 1.2)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = arrowsize*nuv*v4, arrowsize*nuv*v5
    lines!(ax,[x,x+u], [y,y+v], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v5[1]], [y+v,y+v-v5[2]], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v4[1]], [y+v,y+v-v4[2]], color=(color,transparency),linewidth = linewidth)
end

function arrowc!(ax::Axis,x, y, u, v; kwargs...)
    arrow0!(ax,x-u/2,y-v/2,u,v;kwargs...)
end

function arrow2!(ax::Axis,x, y, u, v; arrowsize=0.386, color=:black, transparency=1,linewidth = 1.2)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = arrowsize*nuv*v4, arrowsize*nuv*v5
    lines!(ax,[x,x+u], [y,y+v], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v5[1]], [y+v,y+v-v5[2]], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v4[1]], [y+v,y+v-v4[2]], color=(color,transparency),linewidth = linewidth)
end

function arrowz!(ax::Axis3,x,y,z, s; arrowsize=0.386, color=:black, transparency=1,linewidth = 1.2)
    u=0
    v=s
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = arrowsize*nuv*v4, arrowsize*nuv*v5
    lines!(ax,[x,x],[y,y+u], [z,z+v], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x,x],[y+u,y+u-v5[1]], [z+v,z+v-v5[2]], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x,x],[y+u,y+u-v4[1]], [z+v,z+v-v4[2]], color=(color,transparency),linewidth = linewidth)
end

