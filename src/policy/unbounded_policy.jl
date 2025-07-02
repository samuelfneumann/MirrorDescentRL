"""
    UnBoundedPolicy{V,D} where {
        V<:AbstractVector,
        D<:ExtendedDistributions.ContinuousUnivariateDistribution,
    } <: AbstractContinuousParameterisedPolicy

An `UnBoundedPolicy` is an `IIDPolicy` which has an underlying distribution with unbounded
support.
"""
struct UnBoundedPolicy{
    V<:AbstractVector,
    D<:ExtendedDistributions.ContinuousUnivariateDistribution,
} <: IIDPolicy{V,D}
    _action_min::V
    _action_max::V
    _action_offset::V
    _clip_action::Bool

    function UnBoundedPolicy{V,D}(
        action_min,
        action_max,
        clip_action,
    ) where {V,D}

        dist_min, dist_max = ExtendedDistributions.extrema(D())

        if isfinite(dist_min)
            action_offset = action_min .- dist_min
        elseif isfinite(dist_max)
            action_offset = action_max .- dist_max
        else
            action_offset = zeros(eltype(action_min), size(action_min))
        end

        return new{V,D}(
            action_min,
            action_max,
            action_offset,
            clip_action,
        )
    end

    # Needed for transferring to GPU
    function UnBoundedPolicy{V,D}(
        action_min,
        action_max,
        action_offset,
        clip_action,
    ) where {V,D}
        return new{V,D}(
            action_min,
            action_max,
            action_offset,
            clip_action,
        )
    end
end

function Adapt.adapt_structure(to, b::UnBoundedPolicy{V,D}) where {V,D}
    action_min = Adapt.adapt_structure(to, b._action_min)
    action_max = Adapt.adapt_structure(to, b._action_max)
    action_offset = Adapt.adapt_structure(to, b._action_offset)

    T = typeof(action_max)
    return UnBoundedPolicy{T,D}(action_min, action_max, action_offset, b._clip_action)
end

function UnBoundedPolicy(D, env::AbstractEnvironment; clip_action)
    F = eltype(eltype(action_space(env)))
    return UnBoundedPolicy{Vector{F},D}(env; clip_action=clip_action)
end

function UnBoundedPolicy{V,D}(
    env::AbstractEnvironment; clip_action
) where {V<:AbstractVector,D}
    as = action_space(env)
    action_min = low(as)
    action_max = high(as)

    return UnBoundedPolicy{V,D}(
        action_min,
        action_max,
        clip_action,
    )
end

function (policy::UnBoundedPolicy{V,D})(params::AbstractVector...) where {V,D}
    return D.(params...) .+ policy._action_offset
end

function untransformed_distribution(
    policy::UnBoundedPolicy{V,D}, params::AbstractVector...,
) where {V,D}
    return D.(params...)
end

function (policy::UnBoundedPolicy{V,D})(params::AbstractMatrix...) where {V,D}
    @assert size(params[1], 1) == size(policy._action_offset)[1]
    return D.(params...) .+ policy._action_offset
end

function untransformed_distribution(
    policy::UnBoundedPolicy{V,D}, params::AbstractMatrix...,
) where {V,D}
    @assert size(params[1], 1) == size(policy._action_offset)[1]
    return D.(params...)
end

function (policy::UnBoundedPolicy{V,D})() where {V,D}
    F = eltype(V)
    return convert(D{F}, D()) .+ policy._action_offset
end

function untransformed_distribution(policy::UnBoundedPolicy{V,D}) where {V,D}
    F = eltype(V)
    return convert(D{F}, D())
end

# ####################################################################
# Convenience constructors for a number of UnBoundedPolicy's
# ####################################################################
function NormalPolicy(env::AbstractEnvironment; clip_action)
    return UnBoundedPolicy(Normal, env; clip_action=clip_action)
end

function LaplacePolicy(env::AbstractEnvironment; clip_action)
    return UnBoundedPolicy(Laplace, env; clip_action=clip_action)
end

function LogisticPolicy(env::AbstractEnvironment; clip_action)
    return UnBoundedPolicy(Logistic, env; clip_action=clip_action)
end

function GammaPolicy(env::AbstractEnvironment; clip_action)
    # TODO: check if the gamma distribution calls C code, I think it does, which is why it
    # doesn't work nicely...
    return UnBoundedPolicy(Gamma, env; clip_action=clip_action)
end
# ####################################################################

# ####################################################################
# Log density
# ####################################################################
function logprob(
    p::UnBoundedPolicy,
    actions::AbstractArray{F,3},
    params::AbstractArray{F,2}...;
    sum_=true,
)::AbstractArray{F} where {F}
    lp = underlying_logprob(p, actions, params)
    return sum_ ? sum(lp; dims=1)[1, :, :] : lp
end

function logprob(
    p::UnBoundedPolicy,
    actions::AbstractArray{F,2},
    params::AbstractArray{F,2}...;
    sum_=true,
)::AbstractArray{F} where {F}
    lp = underlying_logprob(p, actions, params)
    return sum_ ? reshape(sum(lp; dims=1), :) : lp
end

function logprob(
    p::UnBoundedPolicy,
    actions::AbstractArray{F,1},
    params::AbstractArray{F,1}...;
    sum_=true,
)::AbstractArray{F} where {F}
    lp = underlying_logprob(p, actions, params)
    return sum_ ? [sum(lp)] : [lp]
end
# ####################################################################

function istransformed(p::UnBoundedPolicy)
    return any(p._action_offset .!= 0f0)
end

function isscaled(p::UnBoundedPolicy)
    return false
end

function isshifted(p::UnBoundedPolicy)
    return any(p._action_offset .!= 0f0)
end

function transform(p::UnBoundedPolicy, samples::AbstractArray{F,3}) where {F}
    n_samples = size(samples, 2)
    action_offset = repeat(p._action_offset, 1, n_samples)
    return samples .+ action_offset
end

function transform(p::UnBoundedPolicy, samples::AbstractArray{F,2}) where {F}
    samples .+ p._action_offset
end

function transform(p::UnBoundedPolicy, samples::AbstractArray{F,1}) where {F}
    samples .+ p._action_offset
end

function untransform(p::UnBoundedPolicy, actions::AbstractArray{F,3}) where {F}
    n_actions = size(actions, 2)
    action_offset = repeat(p._action_offset, 1, n_actions)
    return actions .- action_offset
end

function untransform(p::UnBoundedPolicy, actions::AbstractArray{F,2}) where {F}
    return actions .- p._action_offset
end

function untransform(p::UnBoundedPolicy, actions::AbstractArray{F,1}) where {F}
    return actions .- p._action_offset
end

function clip(p::UnBoundedPolicy, actions::AbstractArray{F,3}) where {F}
    if p._clip_action
        action_min = reshape(clip_min(p._action_min), :, 1)
        action_max = reshape(clip_max(p._action_max), :, 1)
        return clamp.(
            actions,
            action_min,
            action_max
        )
    else
        return actions
    end
end

function clip(p::UnBoundedPolicy, action)
    action_min = clip_min(p._action_min)
    action_max = clip_max(p._action_max)
    return clamp.(
        action,
        action_min,
        action_max
    )
end
