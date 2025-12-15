
include("../src/iPEPS.jl")


Lx = 2
Ly = 2

Latt = XCPeriTria(Lx,Ly)

figsize = (height = (Ly+1)*60*sqrt(3)/3, width = (Lx + 1)*60)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)

# latticescatter!(ax,Latt)
plotLatt!(ax,Latt;site = true,tplevel = (1,))
resize_to_layout!(fig)
display(fig)

neighbor(Latt,1)

save("lattice/figures/XCPeriTria_$(Lx)x$(Ly).pdf",fig)
save("lattice/figures/XCPeriTria_$(Lx)x$(Ly).png",fig)
# issorted.(neighbor(Latt))
# neighbor(Latt,1;issort = false)


