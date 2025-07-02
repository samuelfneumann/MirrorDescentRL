"""
    AbstractPolicy

Represents any policy
"""
abstract type AbstractPolicy end

"""
    Base.rand(p::AbstractPolicy, args...; kwargs...)

See `sample`.
"""
function Base.rand(p::AbstractPolicy, args...; kwargs...)
    return sample(p, args...; kwargs...)
end

function Base.rand(rng::AbstractRNG, p::AbstractPolicy, args...; kwargs...)
    return sample(p, rng, args...; kwargs...)
end

####################################################################
# AbstractParameterisedPolicy
####################################################################

"""
    AbstractParameterisedPolicy <: AbstractPolicy

Represents any parameterized policy
"""
abstract type AbstractParameterisedPolicy <: AbstractPolicy end

"""
    AbstractDiscreteParameterisedPolicy <: AbstractPolicy

Represents any discrete parameterized policy
"""
abstract type AbstractDiscreteParameterisedPolicy <: AbstractParameterisedPolicy end

"""
    AbstractContinuousParameterisedPolicy <: AbstractPolicy

Represents any continuous parameterized policy
"""
abstract type AbstractContinuousParameterisedPolicy <: AbstractParameterisedPolicy end

"""
    continuous(::AbstractPolicy)::Bool

Return whether the policy is a continuous-action policy
"""
function continuous end

"""
    discrete(::AbstractPolicy)::Bool

Return whether the policy is a discrete-action policy
"""
function discrete end

discrete(::AbstractDiscreteParameterisedPolicy) = true
discrete(::AbstractContinuousParameterisedPolicy) = false
function discrete(::AbstractParameterisedPolicy)::Bool
    error("discrete not implemented")
end

continuous(::AbstractDiscreteParameterisedPolicy) = false
continuous(::AbstractContinuousParameterisedPolicy) = true
function continuous(::AbstractParameterisedPolicy)::Bool
    error("continuous not implemented")
end

"""
    sample(
        p::AbstractParameterisedPolicy,
        rng::AbstractRNG,
        model,
        model_θ,
        model_st,
        states;
        [num_samples = 1,
        clip_actions = false],
    )

Return `num_samples` actions from policy `p`.

The `clip_actions` parameter has a few different effects:

- For discrete policies (`SoftmaxPolicy` and `TabularPolicy` types), this parameter is not
    supported
- For continuous policies with infinite support (`UnBoundedPolicy` types), this controls
    whether action samples should be clamped to be withing the valid range of actions, by
    default `true`.
- For continuous policies with finite support (`BoundedPolicy` types), determines whether
    the distribution boundary should be included when sampling actions, which can happen due
    to numerical issues. By default, this is `true`.
"""
function sample end

"""
    sample_with_logprob(
        p::AbstractParameterisedPolicy,
        rng::AbstractRNG,
        model,
        model_θ,
        model_st,
        states;
        [num_samples = 1,
        clip_actions = false],
    )

Sample `num_samples` actions from policy `p` and return the log-probability of sampling these
actions under the policy `p`. See [`sample`](@ref sample) for more information on the
function arguments.

## Numerical Stability

The difference between using this function and using

    julia> actions = sample(
        p, rng, model, model_θ, model_st, states;
        num_samples = num_samples, clip_actions = clip_actions,
    )[1]
    julia> logprob(p, model, model_θ, model_st, states, actions)

Is that the log-probability calculated when using this function is more numerically stable
for some distributions. If actions are sampled along the distribution boundary, for finite
support policy distributions, then this function will return a valid log-probability,
whereas calling `logprob` will result in infinity.
"""
function sample_with_logprob end

"""
    rsample(
        p::AbstractParameterisedPolicy,
        rng::AbstractRNG,
        model,
        model_θ,
        model_st,
        states;
        num_samples = 1,
        clip_actions = false,
    )

Like `sample` but uses the reparameterization trick. See [`sample`](@ref sample) for more
information.
"""
function rsample end

function rsample_with_logprob end

"""
    rsample_with_logprob(
        p::AbstractParameterisedPolicy,
        rng::AbstractRNG,
        model,
        model_θ,
        model_st,
        states;
        num_samples = 1,
        clip_actions = false,
    )

Like `sample_with_logprob` but uses the reparameterization trick
"""

function sample(p::AbstractParameterisedPolicy, rng::AbstractRNG, args...; kwargs...)
    error(
        "sample not implemented for policy of type $(typeof(p)) with args " *
        "::$(typeof(rng)), ::$(typeof(args)), ::$(typeof(kwargs))"
    )
end

function sample(
    p::AbstractParameterisedPolicy,
    model,
    model_θ,
    model_st,
    states;
    num_samples = 1,
    kwargs...
)
    return sample(
        p, Random.GLOBAL_RNG, model, model_θ, model_st, states;
        num_samples = num_samples, kwargs...
    )
end

function rsample(::AbstractParameterisedPolicy, rng::AbstractRNG, args...; kwargs...)
    error("rsample not implemented")
end

function rsample(
    p::AbstractParameterisedPolicy,
    model,
    model_θ,
    model_st,
    states;
    num_samples = 1,
    kwargs...
)
    return rsample(
        p, Random.GLOBAL_RNG, model, model_θ, model_st, states;
        num_samples = num_samples, kwargs...
    )
end

"""
    logprob(
        ::AbstractContinuousParameterisedPolicy,
        f,
        ps,
        st,
        states::AbstractArray,
        actions::AbstractArray,
        sum_::Bool
    )
    logprob(
        ::AbstractDiscreteParameterisedPolicy,
        f,
        ps,
        st,
        states::AbstractArray,
        [actions::AbstractArray]
    )

    Calculate the log-density of actions `actions` in states `states` under the policy
    with model `f` with parameters `ps` and state `st`.
"""
function logprob end

Base.eltype(::AbstractParameterisedPolicy) = error("eltype not implemented")
####################################################################

####################################################################
# AbstractStaticPolicy
####################################################################
"""
    AbstractStaticPolicy <: AbstractPolicy

A policy with static distribution parameters. These parameters *cannot* be learned with
a function approximator and are constant between states.

# Interface

AbstractStaticPolicy subtypes should implement:
    sample(p::AbstractStaticPolicy, rng; num_samples)
    logprob(p::AbstractStaticPolicy, state, action)
"""
abstract type AbstractStaticPolicy <: AbstractPolicy end
