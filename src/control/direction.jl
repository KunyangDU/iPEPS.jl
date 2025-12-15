
struct RIGHT <: AbstractDirection end
struct UP <: AbstractDirection end
struct LEFT <: AbstractDirection end
struct DOWN <: AbstractDirection end

Base.adjoint(::RIGHT) = LEFT()
Base.adjoint(::LEFT) = RIGHT()
Base.adjoint(::UP) = DOWN()
Base.adjoint(::DOWN) = UP()

function _direction(R::AbstractVector)
    @assert norm(R) ≠ 0
    if dot(R,[0,1]) == 0
        if dot(R,[1,0]) > 0
            return RIGHT()
        else
            return LEFT()
        end
    end 
    if dot(R,[1,0]) == 0
        if dot(R,[0,1]) > 0
            return UP()
        else
            return DOWN()
        end
    end
end