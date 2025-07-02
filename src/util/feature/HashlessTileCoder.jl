# Adapted from:
# https://github.com/mkschleg/MinimalRLCore.jl/blob/master/src/features/HashlessTileCoder.jl

"""
    HashlessTileCoder(tiles_per_dim, bounds_per_dim, num_tilings, wrap, offset)

This is a struct for tile-coding raw states to a list of active indices. The constructor
described here uses the helper function `hashlesstilecoder_args` to convert human-readable
variables into the quantities used to compute the tiled indices.

# Arguments
- `tiles_per_dim::Vector{<:Integer}`: A vector describing the number of tiles per dimension
    of the state input. To tile two dimensions of state together with 2 tiles on the first
    dimension and 3 tiles on the second, set `tiles_per_dim = [2, 3]`.
- `bounds_per_dim::Matrix{<:Real}`: A matrix of real values denoting the upper and lower
    bounds of each state dimension. If both state dimensions are bounded in `[0, 1]`, pass
    `bounds_per_dim = [0 0 ; 1 1]`.
- `num_tilings::Integer`: The tilecoder will map state to `num_tilings` indices.
- `wrap::Union{AbstractVector{Bool}, Nothing}`: Nothing, indicating no wrapping, or a vector
    of booleans indicating whether that index of the state input should be wrapped. Setting
    `wrap[k] = true` means that the two ends of `bounds_per_dim[:,k]` are equivalent. For
    example, if `state[k]` describes an angle between 0 and 2π, `wrap[k]` should be set to
    `true`.
- `offset`: A function that describes how each tiling should be offset from the others. The
    recommended setting is to use the default odd-numbered offsets, called asymmetric
    displacement by [Parks and Militzer](https://doi.org/10.1016/S1474-6670(17)54222-6).
"""
struct HashlessTileCoder <: AbstractFeatureCreator
    limits::Matrix{Float32}
    norm_dims::Vector{Float32}
    tiling_dims::Vector{Int32}
    wrap_any_dims::Bool
    offsets::Matrix{Float32}
    tiling_loc::Vector{Int}
    tile_loc::Vector{Int}
    num_features::Int
    num_active_features::Int
    include_bias::Bool
    sum_to_one::Bool

    function HashlessTileCoder(a...; k...)
        l, n, t, w, o, tgl, til, nf, naf, b, sto = _hashlesstilecoder_args(a...; k...)
        new(l, n, t, w, o, tgl, til, nf, naf, b, sto)
    end
end

function (h::HashlessTileCoder)(s; use_onehot=true)
    return use_onehot ? onehot(h, s) : index(h, s)
end

nonzero(fc::HashlessTileCoder) = fc.num_active_features
features(fc::HashlessTileCoder) = fc.num_features
include_bias(fc::HashlessTileCoder) = fc.include_bias
Base.size(fc::HashlessTileCoder) = fc.num_features
index(fc::HashlessTileCoder, s) = _index(fc, s)
onehot(fc::HashlessTileCoder, s::AbstractVector{T}) where {T} = onehot(T, fc, s)

function onehot(T::Type, fc::HashlessTileCoder, s)
    x = zeros(T, fc.num_features)
    elem = fc.sum_to_one ? one(T) / nonzero(fc) : one(T)
    x[_index(fc, s)] .= elem
    return x
end

function onehot(T::Type{<:Integer}, fc::HashlessTileCoder, s)
    x = zeros(T, fc.num_features)
    x[_index(fc, s)] .= one(T)
    return x
end

"""
Constructs a list of active indices from the quantities defined by `hashlesstilecoder_args`.
"""
function _index(fc::HashlessTileCoder, s::AbstractVector)
    if fc.wrap_any_dims
        # wrapping means modding by dim[i] instead of dim[i] + 1
        off_coords = map(x -> floor(Int, x),
                         ((s .- fc.limits[1, :])
                          .* fc.norm_dims
                          .+ fc.offsets)
                         .% fc.tiling_dims)
    else
        # don't need to mod here, because dim[i] + 1 is bigger than the
        # displaced floats
        off_coords = Int.(floor.((s .- fc.limits[1, :]) .* fc.norm_dims .+ fc.offsets))
    end

    return if fc.include_bias
        [1, (fc.tiling_loc .+ off_coords' * fc.tile_loc .+ 2)...]
    else
        fc.tiling_loc .+ off_coords' * fc.tile_loc .+ 1
    end
end

"""
Helper function to make constructing each feature vector later easier
"""
function _hashlesstilecoder_args(
    tiles_per_dim::Vector{<:Integer},
    bounds_per_dim::Matrix{<:Real},
    num_tilings::Integer;
    wrap::Union{AbstractVector{Bool}, Nothing}=nothing,
    offset=n -> collect(1:2:2*n-1), # 1st n odd nums
    include_bias=true,
    sum_to_one=true,
)
    n = length(tiles_per_dim)

    # these normalize the ith input float to be between 0 and dim[i] + 1
    limits = bounds_per_dim
    norm_dims = tiles_per_dim ./ (limits[2, :] .- limits[1, :])

    # wrapping means not adding 1 to the ith dim
    if wrap == nothing
        bonus = ones(Bool, n)
    else
        bonus = .!wrap
    end
    wrap_any_dims = any(.!bonus)
    tiling_dims = tiles_per_dim .+ bonus

    # displacement matrix; default is assymetric displacement a la Parks
    # and Militzer https://doi.org/10.1016/S1474-6670(17)54222-6
    offset_vec = offset(n)
    offsets = (
        offset_vec .* hcat([collect(0:num_tilings-1) for _ in 1:n]...)' ./ num_tilings .% 1
    )

    # these send each displaced float to the proper index
    tiling_loc = collect(0:num_tilings-1) .* prod(tiling_dims)
    tile_loc = [prod(tiling_dims[1:i-1]) for i in 1:n]

    # the total number of indices needed
    num_features = num_tilings * prod(tiling_dims) + include_bias
    num_active_features = num_tilings + include_bias

    return (
        limits, norm_dims, tiling_dims, wrap_any_dims, offsets, tiling_loc, tile_loc,
        num_features, num_active_features, include_bias, sum_to_one,
    )
end
