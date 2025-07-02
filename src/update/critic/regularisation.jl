"""
    AbstractBellmanRegulariser <: AbstractRegulariser

AbstractBellmanRegulariser implements a struct which returns additional terms to add to the
Bellman operator. The Bellman operator 𝕋 is:

  𝕋q(s, a) = r + γq(s', a')

a regulariser adds terms ξ which alters the Bellman operator to a regularised Bellman
operator 𝒯:

  𝒯q(s, a) = r + γ (q(s', a') + ξ)

# References
[1] Nino Vieillard, Tadashi Kozuno, Bruno Scherrer, Olivier Pietquin, Rémi Munos, Matthieu
Geist. Leverage the Average: an Analysis of KL Regularization in RL. Neurips, 2020.

[2] Matthieu Geist, Bruno Scherrer, Olivier Pietquin. A Theory of Regularized Markov
Decision Processes. ICML, 2019.

"""
abstract type AbstractBellmanRegulariser <: AbstractRegulariser end

function regularise end

function regularise(
    reg_state::RegulariserState{<:AbstractBellmanRegulariser},
    states,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
)
    error("regularise not implemented for $(typeof(reg))")
end

function setup(reg::AbstractBellmanRegulariser, args...; kwargs...)
    error(
        "setup not implemented for type $(typeof(reg)) with args $(typeof(args)) " *
        "and kwargs $(typeof(kwargs))"
    )
end

function setup(
    reg::AbstractBellmanRegulariser,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    seed::Integer,
)::RegulariserState{<:AbstractBellmanRegulariser}
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    return setup(reg, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, rng)
end

####################################################################
# KL Regulariser
#
# The KL regulariser refreshes the previous policy parameters in a
# particular way. In order to properly interleave the parameter
# refreshes for proximal mirror descent actor updates and critic
# updates, the actor should be updated first, followed by the critic.
####################################################################
struct KLBellmanRegulariser <: AbstractBellmanRegulariser
    _inv_λ::Float32
    _num_md_updates::Int

    # The number of samples to use for estimating the KL divergence. If an analytic form of
    # the KL divergence exists, that is used and this field is ignored
    _num_samples::Int
    _forward_direction::Bool

    function KLBellmanRegulariser(λ, num_md_updates, num_samples, forward_direction)
        error("KLBellmanRegulariser does not work with the new Proximal update interface")

        @assert λ > 0
        @assert num_md_updates >= 1
        @assert num_samples > 0

        return new(inv(λ), num_md_updates, num_samples, forward_direction)
    end
end

function KLBellmanRegulariser(λ, num_md_updates, forward_direction)
    KLBellmanRegulariser(λ, num_md_updates, 1, forward_direction)
end

function setup(
    reg::KLBellmanRegulariser,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    rng::AbstractRNG,
)::RegulariserState{KLBellmanRegulariser}
    return RegulariserState(
        reg,
        (
            rng = Lux.replicate(rng),
            φ_t = π_θ,      # Policy parameters at the previous step
            st_t = π_st,    # Policy function approximator state at the previous step
            current_update = 1,
        ),
    )
end

function setup(
    reg::KLBellmanRegulariser,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    rng::AbstractRNG,
)::RegulariserState{KLBellmanRegulariser}
    msg = (
        "num_samples should not be modified -- KL is computed exactly in the discrete " *
        "action case"
    )
    @assert reg._num_samples == 1 msg
    return RegulariserState(
        reg,
        (
            φ_t = π_θ,      # Policy parameters at the previous step
            st_t = π_st,    # Policy function approximator state at the previous step
            current_update = 1,
        ),
    )
end

function regularise(
    state::RegulariserState{KLBellmanRegulariser},
    states,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
)

    reg = state._reg
    rng = Lux.replicate(state._state.rng)

    current_update = state._state.current_update
    φ_t, st_t = if current_update == 1
        π_θ, π_st
    else
        state._state.φ_t, state._state.st_t
    end

    kl, π_st, _ = if reg._forward_direction
        kldivergence(
            π, π_f, φ_t, st_t, π_θ, π_st, states; rng=rng, num_samples=reg._num_samples,
        )
    else
        kldivergence(
            π, π_f, π_θ, π_st, φ_t, st_t, states; rng=rng, num_samples=reg._num_samples,
        )
    end

    out = -(reg._inv_λ .* kl)

    next_update = mod(state._state.current_update, reg._num_md_updates) + 1
    return out, π_st, qf_st, RegulariserState(
        reg,
        (
            rng = rng,
            φ_t = φ_t,
            st_t = st_t,
            current_update = next_update,
        )
    )
end

function regularise(
    state::RegulariserState{KLBellmanRegulariser},
    states,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    _actions::AbstractMatrix,            # (action dims, batch_size)
    _actions_log_prob::AbstractVector,   # (batch_size,)
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
)
    return regularise(state, states, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st)
end

function regularise(
    state::RegulariserState{KLBellmanRegulariser},
    states,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
)
    reg = state._reg

    current_update = state._state.current_update
    φ_t, st_t = if current_update == 1
        π_θ, π_st
    else
        state._state.φ_t, state._state.st_t
    end

    kl, π_st, _ = if reg._forward_direction
        kldivergence(π, π_f, φ_t, st_t, π_θ, π_st, states)
    else
        kldivergence(π, π_f, π_θ, π_st, φ_t, st_t, states)
    end

    out = -(reg._inv_λ .* kl)
    next_update = mod(state._state.current_update, reg._num_md_updates) + 1
    return out, π_st, qf_st, RegulariserState(
        reg,
        (
            φ_t = next_update == 1 ? π_θ : φ_t,
            st_t = next_update == 1 ? π_st : st_t,
            current_update = next_update,
        ),
    )
end
####################################################################

####################################################################
# Entropy Regularizer
####################################################################
struct EntropyBellmanRegulariser <: AbstractBellmanRegulariser
    _τ::Float32

    # Not used for discrete actions or when providing the regulariser with actions to use
    # for regularisation
    _num_samples::Int

    function EntropyBellmanRegulariser(τ, n=1)
        @assert τ > 0
        return new(τ, n)
    end
end

function setup(
    reg::EntropyBellmanRegulariser,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    rng::AbstractRNG,
)::RegulariserState{EntropyBellmanRegulariser}
    return RegulariserState(reg, (rng = Lux.replicate(rng),))
end

function setup(
    reg::EntropyBellmanRegulariser,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    rng::AbstractRNG,
)::RegulariserState{EntropyBellmanRegulariser}
    msg = (
        "num_samples should not be modified -- entropy is computed exactly in the " *
        "discrete action case"
    )
    @assert reg._num_samples == 1 msg
    return RegulariserState(reg, NamedTuple())
end

function regularise(
    state::RegulariserState{EntropyBellmanRegulariser},
    states,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
)
    reg = state._reg
    rng = Lux.replicate(state._state.rng)
    actions, lnπ, π_st = sample_with_logprob(
        π, rng, π_f, π_θ, π_st, states; num_samples=reg._num_samples,
    )
    entropy = -dropdims(mean(lnπ; dims=1); dims=1)
    return (reg._τ * entropy), π_st, qf_st, RegulariserState(reg, (rng = rng,))
end

function regularise(
    state::RegulariserState{EntropyBellmanRegulariser},
    states,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    actions::AbstractMatrix,            # (action dims, batch_size)
    actions_log_prob::AbstractVector,   # (batch_size,)
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
)
    # Warning: no checks are done to ensure that the actions_log_prob are the actual log
    # probabilities of actions from policy π
    reg = state._reg
    entropy = -actions_log_prob
    rng = Lux.replicate(state._state.rng)
    return (reg._τ * entropy), π_st, qf_st, RegulariserState(reg, (rng = rng,))
end

function regularise(
    state::RegulariserState{EntropyBellmanRegulariser},
    states,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
)
    reg = state._reg
    H, model_st = entropy(π, π_f, π_θ, π_st, states)

    return (reg._τ .* H), π_st, qf_st, RegulariserState(reg, NamedTuple())
end
####################################################################

####################################################################
# Null Regularizer
####################################################################
struct NullBellmanRegulariser <: AbstractBellmanRegulariser end

function setup(
    reg::NullBellmanRegulariser, args...; kwargs...,
)::RegulariserState{NullBellmanRegulariser}
    return RegulariserState(reg, NamedTuple())
end

function regularise(state::RegulariserState{NullBellmanRegulariser}, args...)
    return 0f0
end
####################################################################
