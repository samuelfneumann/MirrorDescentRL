struct TruncatedPolicy{
    V<:AbstractVector,
   D<:ExtendedDistributions.ContinuousUnivariateDistribution,
} <: IIDPolicy{V,D}
    _action_bias::V
    _action_scale::V
    _action_min::V
    _action_max::V

    # D should not be truncated
    function TruncatedPolicy{V,D}(
        action_min,
        action_max,
    ) where {V,D}
        dist_min, dist_max = ExtendedDistributions.extrema(D())

        dist_min, dist_max = if isfinite(dist_min) && isfinite(dist_max)
            error(
                "expected distribution $D to be unbounded above or below, " *
                "did you mean to use a BoundedPolicy?"
            )
        elseif (dist_min, dist_max) != (-Inf, Inf)
            @warn "TruncatedPolicy only supports distributions with support on " *
                "ℝ, but you've called it on a distribution with " *
                "support ($dist_min, $dist_max). Trying to shift and scale the " *
                "distribution to be within the action bounds."

            if max == Inf
                action_range = (action_max .- action_min)
                bounds = ExtendedDistributions.extrema.(
                    _truncate.(D, dist_min, dist_min .+ action_range),
                )

            elseif min == -Inf
                action_range = (action_max .- action_min)[1]
                bounds = ExtendedDistributions.extrema(
                    _truncate(D, dist_max - action_range, dist_max),
                )

            else
                error("shouldn't be possible to get here")
            end
            dist_min = [b[1] for b in bounds]
            dist_max = [b[2] for b in bounds]
            dist_min, dist_max
        else
            bounds = ExtendedDistributions.extrema.(_truncate.(D, action_min, action_max))
            dist_min = [b[1] for b in bounds]
            dist_max = [b[2] for b in bounds]
            dist_min, dist_max
        end

        action_scale = (action_max .- action_min) ./ (dist_max - dist_min)
        action_bias = -action_scale .* dist_min .+ action_min

        return new{V,D}(
            action_bias,
            action_scale,
            action_min,
            action_max,
        )
    end
end


function TruncatedPolicy(D, env::AbstractEnvironment)
    F = eltype(eltype(action_space(env)))
    return TruncatedPolicy{Vector{F},D}(env)
end

function TruncatedPolicy{V,D}(env::AbstractEnvironment) where {V,D}
    as = action_space(env)
    action_min = low(as)
    action_max = high(as)

    return TruncatedPolicy{V,D}(action_min, action_max)
end

function (policy::TruncatedPolicy{V,D})(params::AbstractVector...) where {V,D}
    dist = _truncate.(D, policy._action_min, policy._action_max, params...)
    return if (
        (any(policy._action_bias .!= 0f0) || any(policy._action_scale .!= 1f0))
    )
        dist .* policy._action_scale .+ policy._action_bias
    else
        dist
    end
end

function untransformed_distribution(
    policy::TruncatedPolicy{V,D}, params::AbstractVector...,
) where {V,D}
    return _truncate.(D, policy._action_min, policy._action_max, params...)
end

function (policy::TruncatedPolicy{V,D})(params::AbstractMatrix...) where {V,D}
    @assert size(params[1], 1) == size(policy._action_scale)[1]
    dist = _truncate.(D, policy._action_min, policy._action_max, params...)
    return if (
        (any(policy._action_bias .!= 0f0) || any(policy._action_scale .!= 1f0))
    )
        dist .* policy._action_scale .+ policy._action_bias
    else
        dist
    end
end

function untransformed_distribution(
    policy::TruncatedPolicy{V,D}, params::AbstractMatrix...,
) where {V,D}
    @assert size(params[1], 1) == size(policy._action_scale)[1]
    return _truncate.(D, policy._action_min, policy._action_max, params...)
end

function (policy::TruncatedPolicy{V,D})() where {V,D}
    F = eltype(V)
    converted_D = typeof(convert(D{F}, D()))
    dist = _truncate(converted_D, policy._action_min, policy._action_max)
    return  dist .* policy._action_scale .+ policy._action_bias
end

function untransformed_distribution(policy::TruncatedPolicy{V,D}) where {V,D}
    F = eltype(V)
    converted_D = typeof(convert(D{F}, D()))
    return _truncate(converted_D, policy._action_min, policy._action_max)
end

function valid_fa(b::TruncatedPolicy, env, fa)
    out = fa(rand(observation_space(env)))
    expected = length(params(D()))
    if !(out isa Tuple) || length(out) != expected
        error("expected approximator to output a $expected-Tuple but got $(typeof(out))")
    end
    if size(out[1])[1] != size(as)[1]
        error("expected approximator to output $(size(as)[1]) values for environment " *
            "$env but got $(size(out[1])[1])")
    end
    return nothing
end

# ####################################################################
# Convenience constructors for a number of BoundedPolicy's
# ####################################################################
function TruncatedNormalPolicy(env::AbstractEnvironment)
    V = eltype(action_space(env))
    return TruncatedPolicy{V,Normal}(env)
end

function TruncatedLaplacePolicy(env::AbstractEnvironment)
    V = eltype(action_space(env))
    return TruncatedPolicy{V,Laplace}(env)
end
####################################################################

function istransformed(p::TruncatedPolicy)
    return any(p._action_bias .!= 0f0) || any(p._action_scale .!= 1f0)
end

function isscaled(p::TruncatedPolicy)
    return any(p._action_scale .!= 1f0)
end

function isshifted(p::TruncatedPolicy)
    return any(p._action_bias .!= 1f0)
end

function transform(p::TruncatedPolicy, samples::AbstractArray{F,3}) where {F}
    n_samples = size(samples, 2)
    action_bias = repeat(p._action_bias, 1, n_samples)
    action_scale = repeat(p._action_scale, 1, n_samples)
    return samples .* action_scale .+ action_bias
end

function transform(p::TruncatedPolicy, samples::AbstractArray{F,2}) where {F}
    samples .* p._action_scale .+ p._action_bias
end

function transform(p::TruncatedPolicy, samples::AbstractArray{F,1}) where {F}
    samples .* p._action_scale .+ p._action_bias
end

function untransform(p::TruncatedPolicy, actions::AbstractArray{F,3}) where {F}
    n_actions = size(actions, 2)
    action_bias = repeat(p._action_bias, 1, n_actions)
    action_scale = repeat(p._action_scale, 1, n_actions)
    untransformed = actions .- action_bias
    return untransformed ./ action_scale
end

function untransform(p::TruncatedPolicy, actions::AbstractArray{F,2}) where {F}
    untransformed = actions .- p._action_bias
    return untransformed ./ p._action_scale
end

function untransform(p::TruncatedPolicy, actions::AbstractArray{F,1}) where {F}
    untransformed = actions .- p._action_bias
    return untransformed ./ p._action_scale
end

function clip(p::TruncatedPolicy, actions::AbstractArray{F,3}) where {F}
    action_min = reshape(clip_min(p._action_min), :, 1)
    action_max = reshape(clip_max(p._action_max), :, 1)
    return clamp.(
        actions,
        action_min,
        action_max
    )
end

function clip(p::TruncatedPolicy, action)
    action_min = clip_min(p._action_min)
    action_max = clip_max(p._action_max)
    return clamp.(
        action,
        action_min,
        action_max
    )
end

function _truncate(D, action_min, action_max, params...)
    dist_min, dist_max = ExtendedDistributions.extrema(D(params...))
    lower = maximum((dist_min, action_min))
    upper = minimum((dist_max, action_max))
    return truncated(D(params...), lower, upper)
end

function _truncate(D, F::Type{<:AbstractFloat}, action_min, action_max, params...)
    dist_min, dist_max = ExtendedDistributions.extrema(D{F}(params...))
    lower = maximum((dist_min, action_min))
    upper = minimum((dist_max, action_max))
    return truncated(D{F}(params...), lower, upper)
end

function logprob(
    p::TruncatedPolicy{<:AbstractVector},
    actions::AbstractArray{F,3},
    params::AbstractArray{F,2}...;
    sum_=true,
)::AbstractArray{F} where {F}
    if size(params[1]) != size(actions)[[1, 3]]
        error("must specify one set of parameters for each action")
    end

    # Construct and reshape policy distributions for broadcasting along action samples
    dist = p(params...)
    dist = reshape(dist, size(dist, 1), 1, size(dist, 2))

    # Calculate log probability, broadcasting along dimension 2, the number of action
    # samples
    lp = logpdf.(dist, actions)
    out = if sum_
        sum(lp; dims=1)[1, :, :]
    else
        lp
    end

    return out
end

function logprob(
    p::TruncatedPolicy{<:AbstractVector},
    actions::AbstractArray{F,2},
    params::AbstractArray{F,2}...;
    sum_=true,
)::AbstractArray{F} where {F}
    if size(params[1]) != size(actions)
        error("must specify one set of parameters for each action")
    end

    # Construct and reshape policy distributions for broadcasting along action samples
    dist = p(params...)

    # Calculate log probability, broadcasting along dimension 2, the number of action
    # samples
    lp = logpdf.(dist, actions)
    out = if sum_
        reshape(sum(lp; dims=1), :)
    else
        reshape(lp, :)
    end

    return out
end

function logprob(
    p::TruncatedPolicy{<:AbstractVector},
    actions::AbstractArray{F,1},
    params::AbstractArray{F,1}...;
    sum_=true,
)::AbstractArray{F} where {F}
    if size(params[1]) != size(actions)
        error("must specify one set of parameters for each action")
    end
    dist = p(params...)
    out = if sum
        [sum(logpdf.(dist, actions))]
    else
        [logpdf.(dist, actions)]
    end
    return out
end
