"""
    IIDPolicy{V,D} where {
        V<:AbstractVector,
        D<:ExtendedDistributions.ContinuousUnivariateDistribution,
    } <: AbstractContinuousParameterisedPolicy

An `IIDPolicy` is a policy which has action dimensions sampled independently and identically
distributed (IID).

We refer to `D` as the _underlying distribution_ of an `IIDPolicy`.
"""
abstract type IIDPolicy{
    V<:AbstractVector, # Used for dispatching based on CPU/GPU device
    D<:ExtendedDistributions.ContinuousUnivariateDistribution,
} <: AbstractContinuousParameterisedPolicy
end

continuous(::IIDPolicy) = true
discrete(::IIDPolicy) = false
Base.eltype(::IIDPolicy{V}) where {V} = V

function ExtendedDistributions.analytical_kl(::IIDPolicy{V,D}) where {V,D}
    analytical_kl(D)
end

function ExtendedDistributions.analytical_entropy(::IIDPolicy{V,D}) where {V,D}
    analytical_entropy(D)
end

function (p::IIDPolicy)(params::AbstractVector...)
    error(
        "constructing distributions from IIDPolicy of type $(typeof(p)) on " *
        "vectors is not implemented",
    )
end

function (p::IIDPolicy)(params::AbstractMatrix...)
    error(
        "constructing distributions from IIDPolicy of type $(typeof(p)) on " *
        "matrices is not implemented",
    )
end

function (p::IIDPolicy)()
    error(
        "constructing distributions from IIDPolicy of type $(typeof(p)) " *
        "is not implemented",
    )
end

# ####################################################################
# Distribution statistics, entropy, KL divergence
# ####################################################################
function extrema(policy::IIDPolicy)
    ex = ExtendedDistributions.extrema.(policy())
    min, max = [e[1] for e in ex], [e[2] for e in ex]
    return min, max
end

# Calculates KL(p || q) where p and q are the same distribution class
# Recall that the KL divergence is invariant under an affine transformation
# https://statproofbook.github.io/P/kl-inv.html
function Distributions.kldivergence(
    dist::IIDPolicy{V,D},
    p_model,
    p_θ,
    p_st,
    q_model,
    q_θ,
    q_st,
    states::AbstractArray;
    rng=Random.default_rng(),
    num_samples=1,
) where {V,D}
    return if analytical_kl(dist)
        _analytic_kldivergence(dist, p_model, p_θ, p_st, q_model, q_θ, q_st, states)
    else
        _estimated_kldivergence(
            dist, rng, p_model, p_θ, p_st, q_model, q_θ, q_st, states;
            num_samples=num_samples,
        )
    end
end

function Distributions.kldivergence(
    dist::IIDPolicy{V,D}, p_params, q_params; rng=Random.default_rng(), num_samples=1,
) where {V,D}
    return if analytical_kl(dist)
        _analytic_kldivergence(dist, p_params, q_params)
    else
        _estimated_kldivergence(dist, rng, p_params, q_params; num_samples=num_samples)
    end
end

# KL divergence when both policies share the same function approximator architecture
function Distributions.kldivergence(
    policy::IIDPolicy{V,D},
    model,
    p_θ,
    p_st,
    q_θ,
    q_st,
    states::AbstractArray;
    rng=Random.default_rng(),
    num_samples=1,
) where {V,D}
    return Distributions.kldivergence(
        policy, model, p_θ, p_st, model, q_θ, q_st, states;
        num_samples=num_samples, rng=rng,
    )
end

function _estimated_kldivergence(
    dist::IIDPolicy{V,D},
    rng::AbstractRNG,
    p_model,
    p_θ,
    p_st,
    q_model,
    q_θ,
    q_st,
    states::AbstractArray;
    num_samples=1,
) where {V,D}
    @warn "I need to re-derive this gradient I think..." maxlog=1
    p_params, p_st = get_params(dist, p_model, p_θ, p_st, states)
    q_params, q_st = get_params(dist, q_model, q_θ, q_st, states)
    action_dims = size(p_params[1])[begin]

    kl = _estimated_kldivergence(
        dist, rng, p_params, q_params; num_samples=num_samples,
    )

    return kl, p_st, q_st
end

function _estimated_kldivergence(
    dist::IIDPolicy{V,D}, rng::AbstractRNG, p_params, q_params; num_samples=1,
) where {V,D}
    @warn "I need to re-derive this gradient I think..." maxlog=1
    action_dims = size(p_params[1])[begin]

    kl = if ndims(p_params[1]) > 1
        # Calculating KL from a batch of states
        batch_size = size(p_params[1])[end]

        actions, p_st = ChainRulesCore.ignore_derivatives(sample(
            dist, rng, p_model, p_θ, p_st, states; num_samples=num_samples,
        ))

        lnp, p_st = logprob(dist, p_model, p_θ, p_st, states, actions)
        lnq, q_st = logprob(dist, q_model, q_θ, q_st, states, actions)

        kl = mean(lnp .- lnq; dims=1)[1, :]
    else
        # Calculating KL from a single state
        actions, p_st = ChainRulesCore.ignore_derivatives(sample(
            dist, rng, p_model, p_θ, p_st, states; num_samples=num_samples,
        ))

        # Unsqueeze batch size
        actions = reshape(actions, action_dims, num_samples, 1)

        lnp, p_st = logprob(dist, p_model, p_θ, p_st, states, actions)
        lnq, q_st = logprob(dist, q_model, q_θ, q_st, states, actions)

        mean(lnp .- lnq)
    end

    return kl
end

function _analytic_kldivergence(
    dist::IIDPolicy,
    p_model,
    p_θ,
    p_st,
    q_model,
    q_θ,
    q_st,
    states::AbstractArray;
)
    p_params, p_st = get_params(dist, p_model, p_θ, p_st, states)
    q_params, q_st = get_params(dist, q_model, q_θ, q_st, states)
    return _analytic_kldivergence(dist, p_params, q_params), p_st, q_st
end

function _analytic_kldivergence(dist::IIDPolicy, p_params, q_params)
    kl = (dist |> kl_function).(p_params..., q_params...)
    kl = ndims(p_params[1]) > 1 ? kl[1, :] : kl
    return kl
end

# Due to action clipping, when most of the probability density is placed near the action
# boundaries (i.e. very low entropy distributions), the estimated entropy may be very
# different from the analytical entropy for some distributions
function Distributions.entropy(
    dist::IIDPolicy{V,D},
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    rng=Random.GLOBAL_RNG,
    num_samples=1,
) where {V,D}
    return if analytical_entropy(dist)
        _analytic_entropy(dist, model, model_θ, model_st, states)
    else
        _estimated_entropy(
            dist, rng, model, model_θ, model_st, states;
            num_samples=num_samples,
        )
    end
end

function _analytic_entropy(
    dist::IIDPolicy{V,D},
    model,
    model_θ,
    model_st,
    states::AbstractArray;
) where {V,D}
    params, model_st = get_params(dist, model, model_θ, model_st, states)
    entropy = _analytic_entropy(dist, params)
    return entropy, model_st
end

function _analytic_entropy(dist::IIDPolicy{V,D}, params) where {V,D}
    entropy = if ndims(params[1]) > 1
        # Calculating entropy from a batch of states
        action_dims = size(params[1])[begin]
        ent = sum(Distributions.entropy.(dist(params...)); dims=1)
        ent[1, :]
    else
        sum(Distributions.entropy.(dist(params...)); dims=1)
    end

    return entropy
end

function _estimated_entropy(
    dist::IIDPolicy{V,D},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples=1,
) where {V,D}
    params, model_st = get_params(dist, model, model_θ, model_st, states)
    entropy = _estimated_entropy(dist, rngm, prams; num_samples=num_samples)
    return entropy, model_st
end

function _estimated_entropy(
    dist::IIDPolicy{V,D}, rng::AbstractRNG, params; num_samples=1,
) where {V,D}
    entropy = if ndims(params[1]) > 1
        # Calculating entropy from a batch of states
        batch_size = size(params[1])[end]
        action_dims = size(params[1])[begin]

        actions, p_st = ChainRulesCore.ignore_derivatives(sample(
            dist, rng, p_model, p_θ, p_st, states; num_samples=num_samples,
        ))

        lnp, model_st = logprob(dist, model, model_θ, model_st, states, actions)

        -mean(lnp; dims=1)[1, :]
    else
        # Calculating entropy from a single state
        actions, p_st = ChainRulesCore.ignore_derivatives(sample(
            dist, rng, p_model, p_θ, p_st, states; num_samples=num_samples,
        ))

        # Unsqueeze batch size
        actions = reshape(actions, action_dims, num_samples, 1)

        lnp, model_st = logprob(dist, model, model_θ, model_st, states, actions)

        -mean(lnp)
    end

    return entropy
end

# Note that not all distributions support this, in which case this function errors
function Distributions.mean(
    p::IIDPolicy{V,D},
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples = 1,
) where {V,D}
    params, model_st = get_params(p, model, model_θ, model_st, states)

    μ = if ndims(params[1]) > 1
        # Sampling actions from a batch of states
        batch_size = size(params[1])[end]

        action_dims = size(params[1])[begin]
        μ = Distributions.mean.(p(params...))
        μ = repeat(μ; inner = (1, num_samples))
        reshape(μ, action_dims, num_samples, batch_size)
    else
        # Sampling actions from a single state
        μ = Distributions.mean.(p(params...))
        repeat(μ, 1, num_samples)
    end

    return clip(p, μ), model_st
end

# Note that not all distributions support this, in which case this function errors
function Distributions.median(
    p::IIDPolicy{V,D},
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples = 1,
) where {V,D}
    params, model_st = get_params(p, model, model_θ, model_st, states)

    m = if ndims(params[1]) > 1
        # Sampling actions from a batch of states
        batch_size = size(params[1])[end]

        action_dims = size(params[1])[begin]
        m = Distributions.median.(p(params...))
        m = repeat(m; inner = (1, num_samples))
        reshape(m, action_dims, num_samples, batch_size)
    else
        # Sampling actions from a single state
        m = Distributions.median.(p(params...))
        repeat(m, 1, num_samples)
    end

    return clip(p, m), model_st
end

# Note that not all distributions support this, in which case this function errors
function Distributions.mode(
    p::IIDPolicy{V,D},
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples = 1,
) where {V,D}
    params, model_st = get_params(p, model, model_θ, model_st, states)

    m = if ndims(params[1]) > 1
        # Sampling actions from a batch of states
        batch_size = size(params[1])[end]

        action_dims = size(params[1])[begin]
        m = Distributions.mode.(p(params...))
        m = repeat(m; inner = (1, num_samples))
        reshape(m, action_dims, num_samples, batch_size)
    else
        # Sampling actions from a single state
        m = Distributions.mode.(p(params...))
        repeat(m, 1, num_samples)
    end

    return clip(p, m), model_st
end
# ####################################################################

# ####################################################################
# Log Density
# ####################################################################
function logprob(
    p::IIDPolicy,
    model,
    model_θ,
    model_st,
    states::AbstractArray{F},
    actions::AbstractArray{F};
    sum_=true,
) where {F}
    params, model_st = get_params(p, model, model_θ, model_st, states)
    return logprob(p, actions, params...; sum_=sum_), model_st
end

function logprob(
    p::IIDPolicy, actions::AbstractArray{F}, params::Tuple; sum_=true,
) where {F}
    return logprob(p, actions, params...; sum_=sum_)
end

"""
    underlying_logprob(:IIDPolicy, actions, params)

Returns the log density of actions from the underlying distribution of an IIDPolicy after
transforming the actions back to samples from the underlying distribution.

This function works with the following combinations of `actions` and `params` sizes:
    - `actions::AbstractArray{F,3} where {F}`
      `params::AbstractArray{F,2} where {F}`
    - `actions::AbstractArray{F,2} where {F}`
      `params::AbstractArray{F,2} where {F}`
    - `actions::AbstractArray{F,1} where {F}`
      `params::AbstractArray{F,1} where {F}`
"""
function underlying_logprob end

@inline function underlying_logprob(
    p::IIDPolicy{<:AbstractVector},
    actions::AbstractArray{F},
    params::Tuple;
)::AbstractArray{F} where {F}
    return underlying_logprob(p, actions, params...)
end

@inline function underlying_logprob(
    p::IIDPolicy{<:AbstractVector},
    actions::AbstractArray{F,3},
    params::AbstractArray{F,2}...;
)::AbstractArray{F} where {F}
    if size(params[1]) != size(actions)[[1, 3]]
        error("must specify one set of parameters for each action")
    end
    samples = untransform(p, actions)
    params = reshape.(params, size.(params, 1), 1, size.(params, 2))
    return (p |> logprob_function).(params..., samples)
end

@inline function underlying_logprob(
    p::IIDPolicy{<:AbstractVector},
    actions::AbstractArray{F,2},
    params::AbstractArray{F,2}...;
)::AbstractArray{F} where {F}
    if size(params[1]) != size(actions)
        error("must specify one set of parameters for each action")
    end
    samples = untransform(p, actions)
    return (p |> logprob_function).(params..., samples)
end

@inline function underlying_logprob(
    p::IIDPolicy{<:AbstractVector},
    actions::AbstractArray{F,1},
    params::AbstractArray{F,1}...;
)::AbstractArray{F} where {F}
    if size(params[1]) != size(actions)
        error("must specify one set of parameters for each action")
    end
    samples = untransform(p, actions)
    return (p |> logprob_function).(params..., samples)
end

function get_params(
    p::IIDPolicy,
    model,
    model_θ,
    model_st,
    states::AbstractArray{F},
) where {F}
    return model(states, model_θ, model_st) # returns out, model_st
end
# ####################################################################

# ####################################################################
# Action clipping and transformations
# ####################################################################
function clip(p::IIDPolicy, actions)
    error("clip(::IIDPolicy, ::$(typeof(actions))) not implemented")
end

# Whether the base distribution must be transformed to cover the action space or not
function istransformed(p::IIDPolicy)::Bool
    error("istransformed not implemented")
end

# transform performs the affine transformation of p on samples from the
# (untransformed) distribution of p
# actions = transform(samples)
function transform(p::IIDPolicy, samples)
    error(
        "transform using IIDPolicy of type $(typeof(p)) is not implemented for type " *
        " $(typeof(samples)) samples",
    )
end

# untransform undoes the affine transformation of p on actions sampled from the
# (transformed) distribution of p
# samples = transform⁻¹(actions) = untransform(actions)
function untransform(p::IIDPolicy, samples)
    error(
        "untransform using IIDPolicy of type $(tyepof(p)) is not implemented for type " *
        " $(typeof(samples)) samples",
    )
end

clip_min(min) = min .* abs.((1f0 .+ sign.(min) .* _fEPSILON))
clip_max(max) = max .* abs.((1f0 .- sign.(max) .* _fEPSILON))
# ####################################################################

function valid_fa(p::IIDPolicy, env, model, model_θ, model_st)
    out, model_st = model(rand(observation_space(env)), model_θ, model_st)

    expected = length(params(p()))

    if !(out isa Tuple) || length(out) != expected
        return (
            valid=false,
            info="expected approximator to output a $expected-Tuple but got $(typeof(out))",
        ), model_st
    end
    if size(out[1])[1] != size(as)[1]
        return (
            valid=false,
            info="expected approximator to output $(size(as)[1]) values for environment " *
                "$env but got $(size(out[1])[1])",
        ), model_st
    end
    return (valid=true, info=""), model_st
end
