"""
    Box{T<:Real, N} <: AbstractSpace{T,N}

n-dimensional set of points of type T between `[l⃗, h⃗]`, where both `l⃗`
and `h⃗` are N-dimensional. The set may be discrete or continuous,
depening on if the `Box` is parameterized with `Integer`s or
`AbstractFloat`s.
"""
struct Box{T<:Real,N} <: AbstractSpace{T,N}
    shape::NTuple{N,Int}
    low::AbstractArray{T,N}
    high::AbstractArray{T,N}

    function Box(shape::NTuple{N,Int}, low::AbstractArray{T,N},
        high::AbstractArray{T,N}) where {T<:Real,N}
        N isa Integer || throw(TypeError(:Box, "N must be an Integer", Integer,
            typeof(N)))

        maximum(low .> high) && error("cannot create Box space with low > high")

        new{T,N}(shape, low, high)
    end
end

function Box{T}(low::Vector, high::Vector) where {T<:Real}
    if length(low) != length(high)
        l = length(low)
        h = length(high)
        error("low and high must have same number of elements but got $l and $h")
    end

    low = convert.(T, low)
    high = convert.(T, high)

    return Box(size(low), low, high)
end

function Box(low::Vector, high::Vector)
    if length(low) != length(high)
        l = length(low)
        h = length(high)
        error("low and high must have same number of elements but got $l and $h")
    end

    low, high = promote(low, high)
    return Box(size(low), low, high)
end

function Box(low::Vector{T}, high::Vector{T}) where {T<:Real}
    if length(low) != length(high)
        l = length(low)
        h = length(high)
        error("low and high must have same number of elements but got $l and $h")
    end
    return Box(size(low), low, high)
end

function Box{T}(low, high) where {T<:Real}
    low = convert(T, low)
    high = convert(T, high)
    return Box((1,), [low], [high])
end

function Box(low::T, high::T) where {T<:Real}
    return Box((1,), [low], [high])
end

Base.ndims(b::Box{T,N}) where {T,N} = N

function Base.rand(
    rng::AbstractRNG,
    b::Box{T,N},
    dims::NTuple{M,Int},
    keepdims=false,
) where {T<:Number,N,M}
    return rand(rng, b, dims...; keepdims = keepdims)
end

function Base.rand(
    rng::AbstractRNG,
    b::Box{T,N},
    d::I,
    dims::I...;
    keepdims=false,
) where {T<:Number,N,I<:Integer}
    dims = tuple(d, dims...)
    # Squeeze dimensions of size 1 if appropriate
    if !keepdims
        dims = Int[dims...]
        dims = filter!(elem->elem!=1, dims)
    end

    return if all((!).(boundedabove(b)) .& (!).(boundedbelow(b)))
        # If action space is unbounded, we sample using the default methods for type T
        rand(rng, T, (size(b)..., dims...))
    elseif any((!).(boundedabove(b)) .| (!).(boundedbelow(b)))
        error("sampling from partially-bounded intervals is not implemented")
    elseif any((!).(bounded(b)))
        error("sampling from spaces with only some unbounded dimensions is not implemented")
    else
        # Reshape for broadcasting
        high_b = reshape(high(b), (size(b)..., 1))
        low_b = reshape(low(b), (size(b)..., 1))

        u = rand(rng, T, size(b)..., prod(dims))
        u = u .* (high_b - low_b) .+ low(b)

        # Reshape to dims batch size
        reshape(u, (size(b)..., dims...))
    end
end

function bounded(b::Box{T,N})::AbstractArray{Bool,N} where
{T<:Real,N}
    boundedbelow(b) .& boundedabove(b)
end

function boundedbelow(b::Box{T,N})::AbstractArray{Bool,N} where
{T<:Real,N}
    return low(b) .> -Inf
end

function boundedabove(b::Box{T,N})::AbstractArray{Bool,N} where
{T<:Real,N}
    return high(b) .< Inf
end

function high(b::Box{T,N})::AbstractArray{T,N} where {T,N}
    b.high
end

function low(b::Box{T,N})::AbstractArray{T,N} where {T,N}
    b.low
end

continuous(::Box)::Bool = true
discrete(::Box)::Bool = false

function Base.contains(b::Box{T,N}, point::AbstractArray{T,N}) where {T<:Real,N}
    return size(point) == size(b) && all(low(b) .<= point .<= high(b))
end

function Base.eltype(b::Box{T,N}) where{T,N}
    return AbstractArray{T,N}
end

function Base.in(point, b::Box)
    return contains(b, point)
end

function Base.size(b::Box)
    return b.shape
end

function Base.size(b::Box, ind)
    return b.shape[ind]
end

function Base.show(io::IO, b::Box)
    l = low(b)
    h = high(b)
    print(io, "Box(low = $l, high = $h)")
end
