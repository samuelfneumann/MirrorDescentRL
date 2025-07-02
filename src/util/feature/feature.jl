abstract type AbstractFeatureCreator end

function (f::AbstractFeatureCreator)(input)
    error("calling $(typeof(f)) is not implemented")
end

export AbstractFeatureCreator, TileCoder, HashlessTileCoder

include("./HashlessTileCoder.jl")

# TODO: TielCoder, GridTiling, HashTiling are all much less efficient than the above
# implementation of `HashlessTileCoder`. They are a bit more flexible though. I am keeping
# them here, because I plan to improve their efficiency in the future, to match to
# performance of the HashlessTileCoder above.
#
# Further, I should update the implementation of these tile coders to use the same offset
# algorithm as in the HashlessTileCoder above. It is much better than random displacement.
include("./tile_coder/AbstractTiling.jl")
include("./tile_coder/TileCoder.jl")
include("./tile_coder/GridTiling.jl")
include("./tile_coder/HashTiling.jl")
