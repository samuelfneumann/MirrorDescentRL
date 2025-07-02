import ChoosyDataLoggers
import ChoosyDataLoggers: @data
# ChoosyDataLoggers.@init

"""
    DiscreteRKL

`DiscreteRKL` implements a discrete-action policy improvement operator which minimizes the
reverse KL-divergence between a learned policy and the Boltzmann distribution over action
values [1]. This operator is equivalent to the policy improvement operator used by SAC [2,
3] when the constructor arguments are chosen appropriately. This update is more generally an
   implementation of the RKL policy improvement operator in [1].

## Updates

This section discusses which update targets are used to update the actor and critic, as well
as some implementation details on how these updates are performed.

### Actor Update

For discrete actions, this algorithm uses the gradient in equation 6 in [1] multiplied by
the entropy scale τ:

    ∇RKL(π, ℬQ) =   τ𝔼_{π} [ τ⁻¹ Q(s, a) - ln(π(a | s))]                              (1)
                =   τ-Σₐ ∇π(a | s) [ τ⁻¹ Q(s, a) - ln(π(a | s))]                      (2)
                =   -Σₐ ∇π(a | s) [ Q(s, a) - τ ln(π(a | s))]                         (3)
where:
    ∇ = ∇_ϕ
    π = π_ϕ
    τ = temperature argument

## References

[1] Greedification Operators for Policy Optimization: Investigating Forward
and Reverse KL Divergences. Chan, A., Silva H., Lim, S., Kozuno, T.,
Mahmood, A. R., White, M. In prepartion. 2021.

[2] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor. Haarnoja, T., Zhou, A., Abbeel, P.,
Levine, S. International Conference on Machine Learning. 2018.

[3] Soft Actor-Critic: Algorithms and Applications. Haarnoja, T.,
Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V.,
Zhu, H., Gupta, A., Abbeel, P., Levine, S. In preparation. 2019.
"""
struct DiscreteRKL <: AbstractActorUpdate
    _temperature::Float32
    _use_baseline::Bool

    function DiscreteRKL(τ, use_baseline)
        return new(τ, use_baseline)
    end
end

function setup(
    up::DiscreteRKL,
    ::AbstractEnvironment,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG,
)::UpdateState{DiscreteRKL}
    return UpdateState(
        up,
        optim,
        (optim = Optimisers.setup(optim, π_θ),),
    )
end

function setup(
    up::DiscreteRKL,
    ::AbstractEnvironment,
    π::SimplexPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Union{Nothing,Optimisers.AbstractRule},
    ::AbstractRNG,
)
    error("cannot use DiscreteRKL with simplex policies")
end

function update(
    st::UpdateState{DiscreteRKL},
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    states::AbstractArray, # Must be >= 2D
)
    up = st._update
    ∇π, π_st, qf_st = _gradient(up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states)

#     p_before, _ = ActorCritic.prob(
#         π, π_f, π_θ, π_st, states,
#     )

    π_optim_state = st._state.optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π))

#     p_after, _ = ActorCritic.prob(
#         π, π_f, π_θ, π_st, states,
#     )

#     log_p_before = log.(p_before)
#     log_p_before[isinf.(log_p_before)] .= 0f0
#     log_p_after = log.(p_after)
#     log_p_after[isinf.(log_p_after)] .= 0f0

#     @data exp norm=mean(mapslices(norm, p_before .- p_after; dims=1))
#     @data exp kl_before_after=mean(sum(p_before .* (log_p_before .- log_p_after); dims=1))
#     @data exp kl_after_before=mean(sum(p_after .* (log_p_after .- log_p_before); dims=1))

    return UpdateState(
        st._update,
        st._optim,
        (optim = π_optim_state,),
    ), π_θ, π_st, qf_st
end

function _gradient(
    up::DiscreteRKL,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    state_batch::AbstractArray, # Must be >= 2D
)
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, state_batch)

    ∇π_θ = gradient(π_θ) do θ
        lnπ, π_st = logprob(π, π_f, θ, π_st, state_batch)
        prob = exp.(lnπ)

        if up._use_baseline
            adv = ChainRulesCore.ignore_derivatives(q .- sum(q .* prob; dims=1))
        else
            adv = q
        end

        scale = if up._temperature != zero(up._temperature)
            # Soft DiscreteRKL
            @. adv - (up._temperature * lnπ)
        else
            # Hard DiscreteRKL
            adv
        end
        scale = ChainRulesCore.ignore_derivatives(scale)

        loss = prob .* scale
        loss = sum(loss; dims=1)
        -gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st
end
