ZZHCmap(Latt::AbstractLattice) = Tuple(sort(1:length(Latt);by = x -> coordinate(Latt,x)[2]))
XCTriamap(Latt::AbstractLattice) = Tuple(1:length(Latt))