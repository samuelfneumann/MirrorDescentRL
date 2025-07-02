"""
    GridTiling{IN<:Number,OUT} <: AbstractTiler{IN<:Number,OUT}

A `GridTiling` represents a grid over some space in ℝ^n and can return either the index of
the tile or a one-hot representation of the tile into which an argument vector in ℝ^n falls.
The `GridTiling` is randomly offset from the origin (see constructor).

A `GridTiling` is parameterized by two types. `IN<:Number` represents the type of inputs
that the tiling can tile code. The tiling can tile code arrays of eltype `IN`. This prevents
unnoticed promotions and casts from affecting performance. The `OUT` type determines what
kind of output the GridTiling outputs. The only condition on the `OUT` type is that the
following functions must work:

1. `zeros(::OUT, ::Int, [::Int])
2. `one(::OUT)`
"""
struct GridTiling{IN<:Number,OUT} <: AbstractTiling{IN,OUT}
    _mindims::Vector{IN}
    _bins::Vector{Int}
    _binlength::Vector{Float32}
    _offsets::Vector{Float32}
    _wrap::Vector{Int}

    function GridTiling{IN,OUT}(
        mindims::Vector,
        maxdims::Vector,
        bins::Vector{Int},
        seed::Integer;
        max_offset = 0.67f0,
        wrap = [],
    ) where {IN<:Number,OUT}
        # Error checking
        if length(mindims) != length(maxdims)
            error("mindims and maxdims must have the same length")
        elseif length(bins) == 0 || length(bins) < length(mindims)
            error("must provide at least one bin per dimension")
        elseif length(bins) > length(mindims)
            error("bin has $(length(bins)) dimensions when state has "*
                "$(length(mindims)) dimensions")
        end

        for dim in wrap
            if dim > length(mindims)
                error("cannot wrap non-existent dimension $dim")
            elseif dim <= 0
                error("cannot wrap negative dimension $dim")
            end
        end

        # Construct the length of bins
        binlength = (maxdims - mindims) ./ bins

        # Get positive, random offset from the origin
        rng = Xoshiro(seed)
        offsets = rand(rng, Float32, length(mindims)) .* binlength
        offsets .*= max_offset

        # Change the sign of certain dimensions' offsets
        signs = zeros(length(mindims))
        signs .-= 1.0f0
        i = rand(rng, Bool, length(mindims))
        signs[i] .= 1.0f0
        offsets .*= signs

        return new{IN,OUT}(mindims, bins, binlength, offsets, wrap)
    end
end

"""
    function GridTiling{OUT}(args...)

Constructor

# Arguments
- `mindims::Vector{IN}`: the minimum value along each dimension that that can be tile coded.
Any values less than this will be assigned to the same bin.
- `maxdims::Vector{IN}`: similar to `mindims` but for maximum values
- `bins::Vector{Int}`: the number of bins along each dimension
- `seed::Integer`: the seed used to generate the random offset from the origin

# Keywords
- `max_offset::Number`: the maximum offset from the origin that the tiling can have along
any dimension in units of bins. For example, `max_offset = 2/3` means that the tiling can
be at most offset from the origin by `2/3` of a bin in any dimension.
- `wrap`: a sequence of bin dimensions along which indices should be wrapped. For
example, if `wrap = [2]`, then dimension `2`'s bins will be wrapped. So if there are a total
of `N` bins along that dimension, and after tile coding an input vector we find that the
vector falls in bin `B`, then the effective bin that is used in the tile coding
representation is `B mod N`.
"""
function GridTiling{OUT}(
    mindims::Vector{IN},
    maxdims::Vector{IN},
    bins::Vector{Int},
    seed::Integer;
    max_offset = 0.67,
    wrap = [],
) where {IN<:Number,OUT}
    return GridTiling{IN,OUT}(mindims, maxdims, bins, seed; max_offset=max_offset, wrap=wrap)
end

function onehot(
    t::GridTiling{IN,OUT}, v::Vector{IN},
)::Vector{OUT} where {IN<:Number,OUT}
    i = index(t, v)
    onehot = zeros(OUT, features(t))
    return setindex!(onehot, one(OUT), i)
end

function onehot(
    t::GridTiling{IN,OUT},
    v::AbstractArray{IN,N},
)::AbstractArray{OUT,N} where {IN<:Number,OUT,N}
    # Get the nonzero indices and reshape so that we can easily perform indexing
    indices = index(t, v)
    s = size(indices)
    indices = reshape(indices, prod(s[1:end])...)

    # Create a matrix of zeros, then use the indices above to change the appropriate
    # non-zero indices to 1
    onehot = zeros(OUT, size(indices)[1], features(t))
    onehot[[CartesianIndex(i, indices[i]) for i = 1:size(indices)[1]]] .= one(OUT)

    # Reshape back to original shape and return
    return reshape(onehot, s..., features(t))
end

function index(t::GridTiling{IN}, v::Vector{IN})::Int where {IN<:Number}
    # Offset the data
    data = v + t._offsets

    # Calculate the index along each dimension that data falls into
    tile = (data - t._mindims) .÷ t._binlength

    if length(t._wrap) != 0
        for dim in t._wrap
            tile[dim] = tile[dim] % t._bins[dim]
        end
    else
        # Clamp the values to be within the tiling bounds. At this point,
        # each coordinate of tile tells which bin along the respective
        # dimension that v falls into. For example, if tile = (1, 4), this
        # means v is in bin 1 along dimension 1 and  bin 4 along dimension 2.
        tile = clamp.(tile, 0.0, t._bins .- 1)
    end

    # Calculate the index of the tile in the flattened tiling into
    # which v falls
    index = tile .* reverse([t._bins[begin:end-1]..., 1])

    # Offset indices to start from 1
    return sum(index) + 1
end

function index(t::GridTiling{IN}, b::Matrix{IN})::Vector{Int} where {IN<:Number}
    # Expand the dimensions of each of the following variables so
    # broadcasting works
    offsets = reshape(t._offsets, size(t._offsets)..., 1)
    mindims = reshape(t._mindims, size(t._mindims)..., 1)
    binlength = reshape(t._binlength, size(t._binlength)..., 1)
    bins = reshape(t._bins, size(t._bins)..., 1)

    # Offset the data
    data = b .+ offsets

    # Calculate the index along each dimension that data falls into
    data .-= mindims
    tile = data .÷ binlength

    if length(t._wrap) != 0
        # Wrap around tiling axes
        for dim in t._wrap
            tile[:, dim] .= tile[:, dim] .% t._bins[dim]
        end
    else
        # Clamp the values to be within the tiling bounds. At this point,
        # each coordinate of tile tells which bin along the respective
        # dimension the current vector in the batch falls into.
        tile = clamp.(tile, 0.0, bins .- 1)
    end

    # Calculate the index of the tile (into the flattened tiling) that
    # each vector in the batch falls into.
    binOffsets = [t._bins[begin:end-1]..., 1]
    binOffsets = reshape(binOffsets, size(binOffsets)..., 1)
    indices = tile .* reverse(binOffsets; dims=1)
    indices = dropdims(sum(indices; dims=1); dims=1)

    return indices .+ 1
end

function index(
    t::GridTiling{IN},
    b::AbstractArray{IN,N},
)::Array{Int,N - 1} where {IN<:Number,N}
    # Convert n-dim array into a matrix
    s = size(b)
    b = reshape(b, prod(s[1:end-1]), s[end])

    # Calculate the tile coding of the matrix
    indices = index(t, b)

    # Reshape back to original shape
    return reshape(indices, s[begin:end-1]...)
end

function features(t::GridTiling)::Int
    return prod(t._bins)
end
