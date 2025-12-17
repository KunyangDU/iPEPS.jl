include("../src/iPEPS.jl")
dataname = "Square/data/test"

Lx = 2
Ly = 2
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 9
params = (J1 = 1.0, J2 = 0.0, h = 0.0)

@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

Sx,Sy,Sz = map(x -> map(y -> data["Obs"][y][x], 1:length(Latt)),1:3)

figsize = (height = (Ly+1)*50, width = (Lx + 1)*50)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)

plotLatt!(ax,Latt;site = true,tplevel = (1,),sitelabel = false,
sitesize = 12*ones(length(Latt))
)

colors = get(colorschemes[:bwr],Sz,(-1/2,1/2))

for i in 1:length(Latt)
    arrowc!(ax,coordinate(Latt,i)...,1.5 *Sx[i],1.5 *Sz[i],linewidth = 3.0,color = colors[i])
end

Colorbar(fig[1,2],colormap = :bwr,colorrange = (-1/2,1/2),label = L"S_z")

resize_to_layout!(fig)
display(fig)

save("Square/figures/pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Square/figures/pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

data["E"]
# Sx
