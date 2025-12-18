
function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, j::Int64, d::AbstractDirection, algo::SimpleUpdate{DynamicSU})
    if algo.scheme.count > algo.scheme.N 
        return _SUupdate!(ψ,O,i,j,d,SimpleUpdate(FullSU(),algo))
    else
        return _SUupdate!(ψ,O,i,j,d,SimpleUpdate(FastSU(),algo))
    end
end

function _SUupdate!(ψ::LGState, O::AbstractTensorMap, i::Int64, algo::SimpleUpdate{DynamicSU})
    if algo.scheme.count > algo.scheme.N 
        return _SUupdate!(ψ,O,i,SimpleUpdate(FullSU(),algo))
    else
        return _SUupdate!(ψ,O,i,SimpleUpdate(FastSU(),algo))
    end
end

function _swap!(ψ::LGState,i::Int64,j::Int64,d::AbstractDirection,algo::SimpleUpdate{DynamicSU})
    if algo.scheme.count > algo.scheme.N 
        return _swap!(ψ,i,j,d,SimpleUpdate(FullSU(),algo))
    else
        return _swap!(ψ,i,j,d,SimpleUpdate(FastSU(),algo))
    end
end
