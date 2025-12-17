include("../src/iPEPS.jl")

Lx = 3
Ly = 1
@load "Honeycomb spin 1/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 3
lsh = 0:0.4:3.2
lsSp = zeros(length(lsh))
for (i,h) in enumerate(lsh)
params = (J1 = -1.0, J3 = 1.0,K = 0.6, D = -3.0, h = -h)

@load "Honeycomb spin 1/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

Sx,Sy,Sz = map(x -> map(y -> data["Obs"][y][x], 1:length(Latt)),1:3)
Sc = -(Sx + Sy + Sz) / sqrt(3)
lsSp[i] = (sum(Sc)) / length(Latt)
end

lsSp

figsize = (height = 200, width = 400)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xlabel = L"h",ylabel = L"\mathbf{S}\cdot \mathbf{\hat{h}}")

lines!(ax,collect(extrema(lsh)),ones(2)/3,color = :grey,linestyle = :dash)
lines!(ax,lsh,lsSp,linewidth = 2,color = :red)

xlims!(ax,extrema(lsh))
ylims!(ax,0,1)

resize_to_layout!(fig)
display(fig)

save("Honeycomb spin 1/figures/MH_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsh)).png",fig)
save("Honeycomb spin 1/figures/MH_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsh)).pdf",fig)

# data["E"]
# sum(Sz) / length(Latt)