"""
    BoundedPolicy{V,D} where {
        V<:AbstractVector,
        D<:ExtendedDistributions.ContinuousUnivariateDistribution,
    } <: AbstractContinuousParameterisedPolicy

A `BoundedPolicy` is an `IIDPolicy` which has an underlying distribution with bounded
support.
"""
struct BoundedPolicy{
    V<:AbstractVector,
    D<:ExtendedDistributions.ContinuousUnivariateDistribution,
} <: IIDPolicy{V,D}
    _action_bias::V
    _action_scale::V
    _action_min::V
    _action_max::V

    # TODO: maybe this can be an outer constructor
    function BoundedPolicy{V,D}(action_min, action_max) where {V<:AbstractVector,D}
        dist_min, dist_max = ExtendedDistributions.extrema(D())

        action_scale = (action_max .- action_min) / (dist_max - dist_min)
        action_bias = -action_scale .* dist_min + action_min

        return BoundedPolicy{V,D}(action_bias, action_scale, action_min, action_max)
    end

    function BoundedPolicy{V,D}(
        action_bias, action_scale, action_min, action_max,
    ) where {V<:AbstractVector,D}
        return new{V,D}(action_bias, action_scale, action_min, action_max)
    end
end

function Adapt.adapt_structure(to, b::BoundedPolicy{V,D}) where {V,D}
    action_bias = Adapt.adapt_structure(to, b._action_bias)
    action_scale = Adapt.adapt_structure(to, b._action_scale)
    action_min = Adapt.adapt_structure(to, b._action_min)
    action_max = Adapt.adapt_structure(to, b._action_max)

    T = typeof(action_max)
    return BoundedPolicy{T,D}(action_bias, action_scale, action_min, action_max)
end

function BoundedPolicy(D, env::AbstractEnvironment)
    F = eltype(rand(action_space(env)))
    return BoundedPolicy{Vector{F},D}(env)
end

function BoundedPolicy{V,D}(env::AbstractEnvironment) where {V,D}
    as = action_space(env)
    action_min = low(as)
    action_max = high(as)

    return BoundedPolicy{V,D}(action_min, action_max)
end

function (policy::BoundedPolicy{V,D})(params::AbstractVector...) where {V,D}
    return D.(params...) .* policy._action_scale .+ policy._action_bias
end

function untransformed_distribution(
    policy::BoundedPolicy{V,D}, params::AbstractVector...,
) where {V,D}
    return D.(params...)
end

function (policy::BoundedPolicy{V,D})(params::AbstractMatrix...) where {V,D}
    @assert size(params[1], 1) == size(policy._action_scale)[1]
    return D.(params...) .* policy._action_scale .+ policy._action_bias
end

function untransformed_distribution(
    policy::BoundedPolicy{V,D}, params::AbstractMatrix...,
) where {V,D}
    return D.(params...)
end

function (policy::BoundedPolicy{V,D})() where {V,D}
    F = eltype(V)
    return convert(D{F}, D()) .* policy._action_scale .+ policy._action_bias
end

function untransformed_distribution(policy::BoundedPolicy{V,D}) where {V,D}
    F = eltype(V)
    return convert(D{F}, D())
end

# ####################################################################
# Convenience constructors for a number of BoundedPolicy's
# ####################################################################
function BetaPolicy(env::AbstractEnvironment)
    return BoundedPolicy(Beta, env)
end

function KumaraswamyPolicy(env::AbstractEnvironment)
    return BoundedPolicy(Kumaraswamy, env)
end

function LogitNormalPolicy(env::AbstractEnvironment)
    return BoundedPolicy(LogitNormal, env)
end

function ArctanhNormalPolicy(env::AbstractEnvironment)
    return BoundedPolicy(ArctanhNormal, env)
end
####################################################################

# ####################################################################
# Log Density
# ####################################################################
function logprob(
    p::BoundedPolicy,
    actions::AbstractArray{F,3},
    params::AbstractArray{F,2}...;
    sum_=true,
)::AbstractArray{F} where {F}
    lp = underlying_logprob(p, actions, params) .- log.(p._action_scale)
    return sum_ ? sum(lp; dims=1)[1, :, :] : lp
end

function logprob(
    p::BoundedPolicy,
    actions::AbstractArray{F,2},
    params::AbstractArray{F,2}...;
    sum_=true,
)::AbstractArray{F} where {F}
    lp = underlying_logprob(p, actions, params) .- log.(p._action_scale)
    return sum_ ? reshape(sum(lp; dims=1), :) : lp
end

function logprob(
    p::BoundedPolicy,
    actions::AbstractArray{F,1},
    params::AbstractArray{F,1}...;
    sum_=true,
)::AbstractArray{F} where {F}
    lp = underlying_logprob(p, actions, params) .- log.(p._action_scale)
    return sum_ ? [sum(lp)] : [lp]
end

# ####################################################################
# Additional implementations of rsamples
# ####################################################################
function rsample(
    p::BoundedPolicy{V,Beta},
    rng::AbstractRNG,
    model,
    model_θ,
    model_state,
    states::AbstractArray;
    num_samples = 1,
) where {V}
    error(
        "rsample not implemented for Beta, see implementation in PyTorch once you " *
        "want to try implementing this, they follow these papers:\n" *
        "- https://arxiv.org/abs/1806.01851 \n" *
        "- https://papers.nips.cc/paper/2018/file/92c8c96e4c37100777c7190b76d28233-Paper.pdf"
    )
end
# ####################################################################

# ####################################################################
# Beta and Kumaraswamy need their samples to be clipped due to
# numerical issues. They can quickly concentrate most probability
# density on their support boundaries, and then we end up rounding
# samples to the support boundaries. When calculating the log
# density, we end up getting -Inf
# ####################################################################
function sample(
    p::IIDPolicy{V,D},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray{F};
    clip_actions=true,
    kwargs...,
) where {V,D<:Union{Kumaraswamy,Beta},F}
    return _sample(
        p, rng, model, model_θ, model_st, states; clip_actions = clip_actions, kwargs...,
    )
end
# ####################################################################

function istransformed(p::BoundedPolicy)
    return any(p._action_bias .!= 0f0) || any(p._action_scale .!= 1f0)
end

function isscaled(p::BoundedPolicy)
    return any(p._action_scale .!= 1f0)
end

function isshifted(p::BoundedPolicy)
    return any(p._action_bias .!= 1f0)
end

function transform(p::BoundedPolicy, samples::AbstractArray{F,3}) where {F}
    n_samples = size(samples, 2)
    action_bias = repeat(p._action_bias, 1, n_samples)
    action_scale = repeat(p._action_scale, 1, n_samples)
    return samples .* action_scale .+ action_bias
end

function transform(p::BoundedPolicy, samples::AbstractArray{F,2}) where {F}
    samples .* p._action_scale .+ p._action_bias
end

function transform(p::BoundedPolicy, samples::AbstractArray{F,1}) where {F}
    samples .* p._action_scale .+ p._action_bias
end

function untransform(p::BoundedPolicy, actions::AbstractArray{F,3}) where {F}
    n_actions = size(actions, 2)
    action_bias = repeat(p._action_bias, 1, n_actions)
    action_scale = repeat(p._action_scale, 1, n_actions)
    untransformed = actions .- action_bias
    return untransformed ./ action_scale
end

function untransform(p::BoundedPolicy, actions::AbstractArray{F,2}) where {F}
    untransformed = actions .- p._action_bias
    return untransformed ./ p._action_scale
end

function untransform(p::BoundedPolicy, actions::AbstractArray{F,1}) where {F}
    untransformed = actions .- p._action_bias
    return untransformed ./ p._action_scale
end

function clip(p::BoundedPolicy, actions::AbstractArray{F,3}) where {F}
    action_min = reshape(clip_min(p._action_min), :, 1)
    action_max = reshape(clip_max(p._action_max), :, 1)
    return clamp.(
        actions,
        action_min,
        action_max
    )
end

function clip(p::BoundedPolicy, action)
    action_min = clip_min(p._action_min)
    action_max = clip_max(p._action_max)
    return clamp.(
        action,
        action_min,
        action_max
    )
end
