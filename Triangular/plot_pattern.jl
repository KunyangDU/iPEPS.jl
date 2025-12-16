include("../src/iPEPS.jl")

Lx = 3
Ly = 3
@load "Triangular/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 3
# params = (J1 = 1.0, J2 = 0.0, h = 5.0)
params = (Jxy = 1.0, Jz = 1.68, h = 0.0)

@load "Triangular/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

Sx,Sy,Sz = map(x -> map(y -> data["Obs"][y][x], 1:length(Latt)),1:3)

figsize = (height = (Ly+1)*100, width = (Lx + 1)*100)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)

plotLatt!(ax,Latt;site = true,tplevel = (1,),sitelabel = false,
sitesize = 12*ones(length(Latt))
)

colors = get(colorschemes[:berlin],Sz,(-1/2,1/2))

for i in 1:length(Latt)
    arrowc!(ax,coordinate(Latt,i)...,1.5 *Sx[i],1.5 *Sy[i],linewidth = 3.0,color = colors[i])
end

Colorbar(fig[1,2],colormap = :berlin,colorrange = (-1/2,1/2),label = L"S_z")

resize_to_layout!(fig)
display(fig)

save("Triangular/figures/pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Triangular/figures/pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

data["E"]
sum(Sz) / length(Latt)