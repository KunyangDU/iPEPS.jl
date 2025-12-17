include("../src/iPEPS.jl")

Lx = 3
Ly = 1
@load "Honeycomb spin 1/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 6
h = 2.4

params = (J1 = -1.0, J3 = 1.0, K = 0.6, D = -3.0, h = h, θ = pi/18)

@load "Honeycomb spin 1/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

Sx,Sy,Sz = map(x -> map(y -> data["Obs"][y][x], 1:length(Latt)),1:3)
proj = - sin(params.θ) * [1,1,-2] / sqrt(2) + cos(params.θ) * [1,1,1]/sqrt(3)
Sp = sum(hcat(Sx,Sy,Sz) * proj,dims = 2)
figsize = (height = (Ly+1)*100, width = (Lx + 1)*100)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)

plotLatt!(ax,Latt;site = true,tplevel = (1,),sitelabel = false,
sitesize = 12*ones(length(Latt))
)

colors = get(colorschemes[:bwr],Sp,(-1,1))

for i in 1:length(Latt)
    arrowc!(ax,coordinate(Latt,i)...,0.0,0.8 *Sp[i],linewidth = 3.0,color = colors[i])
end

Colorbar(fig[1,2],colormap = :bwr,colorrange = (-1,1),label = L"S_z")

resize_to_layout!(fig)
display(fig)

save("Honeycomb spin 1/figures/pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Honeycomb spin 1/figures/pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

data["E"]
# Sx
