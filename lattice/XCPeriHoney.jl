
include("../src/iPEPS.jl")


Lx = 1
Ly = 1

Latt = XCPeriHoneycomb(Lx,Ly)

# figsize = (height = (Ly+1)*80*sqrt(3)/3, width = (Lx + 1)*80)
figsize = getfigsize(Latt,40)
fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)

# latticescatter!(ax,Latt)
plotLatt!(ax,Latt;site = true,tplevel = (1,))
resize_to_layout!(fig)
display(fig)

save("lattice/figures/XCPeriHoney_$(Lx)x$(Ly).pdf",fig)
save("lattice/figures/XCPeriHoney_$(Lx)x$(Ly).png",fig)
# issorted.(neighbor(Latt))

neighbor_pbc(Latt)