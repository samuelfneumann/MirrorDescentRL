struct ObservationFunc{
    E<:AbstractEnvironment,F1,F2,
} <: AbstractEnvironmentObservationWrapper
    env::E
    f::F1
    obsspace::F2
end

observation_space(o::ObservationFunc) = o.obsspace(observation_space(wrapped(o)))
observation(o::ObservationFunc, obs) = o.f(obs)
wrapped(o::ObservationFunc) = o.env

"""
GridTileCoderObservationFunc uses a `TileCoder` with `GridTilings` to tile code observations
"""
function GridTileCoderObservationFunc(
    T::Type, env, bins::Vector{<:Integer}, seed; sum_to_one=true, bias=true,
    max_offset=0.67f0, use_onehot=true,
)
    mindims = ActorCritic.low(observation_space(env))
    maxdims = ActorCritic.high(observation_space(env))
    bins = repeat(bins', size(mindims, 1))

    feature_creator = ActorCritic.GridTileCoder(
        T, mindims, maxdims, bins, seed; sum_to_one, bias, max_offset,
    )

    f = x -> feature_creator(x; use_onehot)
    obsspace = if use_onehot
        low = zeros(T, ActorCritic.features(feature_creator))
        high = ones(T, ActorCritic.features(feature_creator))
        _ -> Box(low, high)
    else
        _ -> Discrete(only(size(feature_creator)) + include_bias(feature_creator))
    end

    return ObservationFunc(env, f, obsspace)
end

function GridTileCoderObservationFunc(
    env, bins::Vector{<:Integer}, seed; kwargs...
)
    return GridTileCoderObservationFunc(Float32, env, bins, seed; kwargs...)
end

"""
HashlessTileCoderObservationFunc uses a `HashlessTileCoder` to tile code observations
"""
function HashlessTileCoderObservationFunc(
    T::Type, env, bins::Vector{<:Integer}, tilings::Integer; sum_to_one=true,
    include_bias=true, wrap=nothing, use_onehot=true,
)
    mindims = ActorCritic.low(observation_space(env))
    maxdims = ActorCritic.high(observation_space(env))
    bounds = vcat(mindims', maxdims')

    feature_creator = HashlessTileCoder(
        bins, bounds, tilings; wrap, include_bias, sum_to_one,
    )

    f = x -> feature_creator(x; use_onehot)
    obsspace = if use_onehot
        low = zeros(T, ActorCritic.features(feature_creator))
        high = ones(T, ActorCritic.features(feature_creator))
        _ -> Box(low, high)
    else
        _ -> Discrete(only(size(feature_creator)) + include_bias(feature_creator))
    end

    return ObservationFunc(env, f, obsspace)
end

function HashlessTileCoderObservationFunc(
    env, bins::Vector{<:Integer}, tilings::Integer; kwargs...
)
    return HashlessTileCoderObservationFunc(Float32, env, bins, tilings; kwargs...)
end
