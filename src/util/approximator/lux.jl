####################################################################
# Concat Layer
####################################################################
struct Concat{N} <: Lux.AbstractLuxLayer end

Concat(n::Int) = Concat{n}()

Lux.initialparameters(rng::AbstractRNG, ::Concat) = NamedTuple()
Lux.parameterlength(::Concat) = 0
Lux.initialstates(rng::AbstractRNG, ::Concat) = NamedTuple()
Lux.statelength(::Concat) = 0

(::Concat{1})(args, ps, st) = reduce(vcat, args), st
(::Concat{2})(args, ps, st) = reduce(hcat, args), st
(::Concat{N})(args, ps, st) where {N} = cat(args...; dims=N), st
####################################################################

const LuxModel = Union{
    Concat,
    Lux.Chain,
    Lux.Dense,
    Lux.SkipConnection,
    Lux.Parallel,
    Lux.PairwiseFusion,
    Lux.SamePad,
    Lux.Conv,
    Lux.AdaptiveMaxPool,
    Lux.AdaptiveMeanPool,
    Lux.GlobalMaxPool,
    Lux.GlobalMeanPool,
    Lux.MaxPool,
    Lux.MeanPool,
    Lux.Dropout,
    Lux.AlphaDropout,
    Lux.LayerNorm,
    Lux.BatchNorm,
    Lux.InstanceNorm,
    Lux.GroupNorm,
    Lux.Upsample,
}

####################################################################
# Extra Functions
####################################################################
"""
    polyak(β::AbstractFloat, dest, src)

Compute the polyak average of the weights of src with the weights of dest using

    new_weights = β * src + (1 - β) * dest
"""
function polyak(tpl::NamedTuple, β::AbstractFloat, dest, src)::NamedTuple
    out = treemap((dest, src) -> _polyak(β, dest, src), dest, src)
    return out
end

function polyak!(β::AbstractFloat, dest, src)
    treemap!((dest, src) -> _polyak(β, dest, src), dest, src)
    return nothing
end

function _polyak(β, dest, src)
    return if β == one(β)
        dest .= src
    elseif β == zero(β)
    else
        dest .= (β .* src .+ (one(β) - β) .* dest)
    end
end


polyak(β::AbstractFloat, dest, src) = polyak(NamedTuple(), β, dest, src)
set(dest, src) = polyak(1f0, dest, src)

# Override train mode function
function train(::LuxModel, model_st::NamedTuple)
    return Lux.trainmode(model_st)
end

# Override eval mode function
function eval(::LuxModel, model_st::NamedTuple)
    return Lux.testmode(model_st)
end

function setup(rng::AbstractRNG, model::LuxModel)
    return Lux.setup(Lux.replicate(rng), model)
end
####################################################################
