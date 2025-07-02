"""
	AbstractSpace{T, N} where {T<:Number}

Some space of numbers with `N` dimensions
"""
abstract type AbstractSpace{T<:Number,N} end

function Base.rand(
    s::AbstractSpace,
    d::I,
    dims::I...;
    keepdims=false,
) where {I<:Integer}
    return rand(Random.GLOBAL_RNG, s, d, dims...; keepdims=keepdims)
end

function Base.rand(
    b::AbstractSpace,
    dims::NTuple{M,I},
    keepdims=false,
) where {M,I<:Integer}
    return rand(Random.GLOBAL_RNG, b, dims...; keepdims = keepdims)
end

function Base.rand(s::AbstractSpace; keepdims=false)
    return rand(Random.GLOBAL_RNG, s; keepdims=keepdims)
end

function Base.rand(rng::AbstractRNG, s::AbstractSpace; keepdims=false)
    return rand(rng, s, 1; keepdims=keepdims)
end

"""
    bounded(s::AbstractSpace)::AbstractArray{Bool, N}

Return a boolean array indicating which dimensions of the space are
bounded
"""
function bounded(s::AbstractSpace)::AbstractArray{Bool,N}
    error("bounded not implemented for Abstractspace $(typeof(s))")
end

"""
    high(s::AbstractSpace)::AbstractArray{T, N}

Return the inclusive upper bound of the space
"""
function high(s::AbstractSpace)::AbstractArray{T,N}
    error("high not implemented for AbstractSpace $(typeof(s))")
end

"""
    low(s::AbstractSpace)::AbstractArray{T, N}

Return the inclusive lower bound of the space
"""
function low(s::AbstractSpace)::AbstractArray{T,N}
    error("low not implemented for AbstractSpace $(typeof(s))")
end

function Base.in(item, s::AbstractSpace)::Bool
    error("Base.in not implemented for AbstractSpace $(typeof(s))")
end

function Base.size(s::AbstractSpace)::Tuple{<:Integer}
    error("size not implemented for AbstractSpace $(typeof(s))")
end

function Base.ndims(s::AbstractSpace{T,N}) where {T,N}
    return N
end

function Base.show(io::IO, a::AbstractSpace)
    error("show not implemented for AbstractSpace $(typeof(s))")
end

function continuous(s::AbstractSpace)::Bool
    error("continuous not implemented for AbstractSpace $(typeof(s))")
end

function discrete(s::AbstractSpace)::Bool
    error("discrete not implemented for AbstractSpace $(typeof(s))")
end

# These two functions determine if a point is in a space. By default, we assume that a point
# is not in a space. Spaces should override this function to determine if a point is in a
# space.
Base.contains(::AbstractSpace, point) = false

function Base.eltype(::AbstractSpace{T,N}) where{T,N}
    return T
end
