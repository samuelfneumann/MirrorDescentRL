"""
N-dimensional discrete set of points of type T, where each dimension
is between `[1⃗, n⃗]`. `n⃗` itself is N-dimensional.
"""
struct Discrete{T<:Integer,N} <: AbstractSpace{T,N}
    shape::NTuple{N,Int}
    n::Array{T,N}

    function Discrete{T,N}(shape::NTuple{N,Int},
        n::AbstractArray{T,N}) where {T<:Integer,N}

        N isa Integer || throw(TypeError(:Box, "N must be an Integer", Integer,
            typeof(N)))

        new(shape, n)
    end
end

function Discrete{T}(n::Integer) where {T<:Integer}
    size = (1,)
    n = [n]

    return Discrete{T,1}(size, convert.(T, n))
end

function Discrete(n::AbstractArray{T,N}) where {T<:Integer,N}
    Discrete{T,N}(size(n), n)
end

function Discrete(n::T) where {T<:Integer}
    Discrete{T,1}((1,), [n])
end

Base.ndims(d::Discrete{T,N}) where {T,N} = N
Base.eltype(d::Discrete{T,N}) where {T,N} = AbstractArray{T,N}

function Base.rand(
    rng::AbstractRNG,
    d::Discrete{T,N},
    dims::NTuple{M,Int},
    keepdims=false,
) where {T<:Number,N,M}
    return rand(rng, d, dims...; keepdims=keepdims)
end

function Base.rand(
    rng::AbstractRNG,
    d::Discrete{T, N},
    dims::Int...;
    keepdims=false,
) where {T<:Integer,N}
    # Squeeze dimensions of size 1 if appropriate
    if !keepdims
        dims = Int[dims...]
        dims = filter!(elem->elem!=1, dims)
    end
    u = rand(rng, T, prod((dims..., size(d)...)))
    u = reshape(u, size(d)..., dims...)
    return mod.(u, high(d)) .+ low(d)
end

function high(d::Discrete{T,N})::AbstractArray{T,N} where {T<:Integer,N}
    d.n
end

function low(d::Discrete{T,N})::AbstractArray{T,N} where {T<:Integer,N}
    ones(T, d.shape)
end

function bounded(d::Discrete{T,N})::AbstractArray{Bool,N} where {T<:Integer,N}
    d.n .< Inf
end

function continuous(::Discrete)::Bool
    return false
end

function discrete(::Discrete)::Bool
    return true
end

function Base.contains(d::Discrete{T,N}, point::AbstractArray{T,N}) where {T<:Integer,N}
    return size(d) == size(point) && all(low(d) .<= point .<= high(d))
end

function Base.contains(d::Discrete{T,1}, point::T) where {T<:Integer}
    return all(low(d) .<= point .<= high(d))
end

function Base.size(d::Discrete)
    return d.shape
end

function Base.size(d::Discrete, ind)
    return d.shape[ind]
end

function Base.show(io::IO, d::Discrete)
	print(io, "Discrete($(d.n...))")
end
