include("../src/iPEPS.jl")

Lx = 2
Ly = 2
@load "Square/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 4
lsh = 0:0.2:4.6
lsSp = zeros(length(lsh))
for (i,h) in enumerate(lsh)
params = (J1 = 1.0, J2 = 0.0, h = h)

@load "Square/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

Sx,Sy,Sz = map(x -> map(y -> data["Obs"][y][x], 1:length(Latt)),1:3)
lsSp[i] = (sum(Sz)) / length(Latt)
end

lsSp

figsize = (height = 200, width = 400)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
yticks = 0:0.1:0.5,
xlabel = L"h",ylabel = L"\mathbf{S}\cdot \mathbf{\hat{h}}")

# lines!(ax,collect(extrema(lsh)),ones(2)/6,color = :grey,linestyle = :dash)
lines!(ax,lsh,lsSp,linewidth = 2,color = :red)

xlims!(ax,0,4.6)
ylims!(ax,0,0.51)

resize_to_layout!(fig)
display(fig)

save("Square/figures/MH_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsh)).png",fig)
save("Square/figures/MH_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsh)).pdf",fig)

# data["E"]
sum(Sz) / length(Latt)
lsSp