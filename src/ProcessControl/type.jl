abstract type AbstractDirection end

struct RIGHT <: AbstractDirection end
struct UP <: AbstractDirection end
struct LEFT <: AbstractDirection end
struct DOWN <: AbstractDirection end

Base.adjoint(::RIGHT) = LEFT()
Base.adjoint(::LEFT) = RIGHT()
Base.adjoint(::UP) = DOWN()
Base.adjoint(::DOWN) = UP()

