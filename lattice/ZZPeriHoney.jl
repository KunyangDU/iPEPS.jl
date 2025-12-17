
include("../src/iPEPS.jl")

Lx = 2
Ly = 2
Latt = ZZPeriHoneycomb(Lx,Ly)
figsize = getfigsize(Latt)


fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)

# latticescatter!(ax,Latt)
plotLatt!(ax,Latt;site = true,tplevel = (1,))

resize_to_layout!(fig)
display(fig)

neighbor(Latt,1)

save("lattice/figures/ZZPeriHoneycomb_$(Lx)x$(Ly).pdf",fig)
save("lattice/figures/ZZPeriHoneycomb_$(Lx)x$(Ly).png",fig)


