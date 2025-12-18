
include("../src/iPEPS.jl")
Lx = 2
Ly = 2
Latt = PeriKagome(Lx,Ly)

figsize = getfigsize(Latt,40)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)

# latticescatter!(ax,Latt)
plotLatt!(ax,Latt;site = true,tplevel = (1,))
resize_to_layout!(fig)
display(fig)

neighbor(Latt,1)

save("lattice/figures/PeriKagome_$(Lx)x$(Ly).pdf",fig)
save("lattice/figures/PeriKagome_$(Lx)x$(Ly).png",fig)
# ZZHCmap(Latt)
Map = PeriKagomeMapping(Latt)
# _Kagome_index(Latt)
# Latt[3]

findpath(Map.auxLatt,(3,[0,0]),(6,[0,0]),Map.banbonds)
