"""
DiscreteFKL implements a policy improvement operator which minimizes the forward
KL-divergence between a learned policy and the Boltzmann distribution over action values
[1]. This update is an implementation of the discrete-action FKL policy improvement operator
in [1].

## Updates

Because this implementation solely follows [1], the temperature τ is the entropy scale,
rather than a reward scale as in [2, 3].

### Actor Update

For discrete actions, this algorithm uses the gradient in equation 7 in [1]:

    ∇FKL(π, ℬQ) = - Σₐ ℬQ(a | s) ∇ln(π(a | s))                                          (1)

where:
    ∇ = ∇_ϕ
    π = π_π

When the temperature τ is 0, the update becomes the hard FKL update, found in equation 9 in
[1]:

    ∇HardFKL(π, ℬQ) = - ∇ln(π( argmaxₐ q(s, a) | s))                                    (2)

where:
    ∇ = ∇_ϕ
    π = π_π
    q = action-value function approximation

## References
[1] Greedification Operators for Policy Optimization: Investigating Forward
and Reverse KL Divergences. Chan, A., Silva H., Lim, S., Kozuno, T.,
Mahmood, A. R., White, M. 2021.

[2] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor. Haarnoja, T., Zhou, A., Abbeel, P.,
Levine, S. International Conference on Machine Learning. 2018.

[3] Soft Actor-Critic: Algorithms and Applications. Haarnoja, T.,
Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V.,
Zhu, H., Gupta, A., Abbeel, P., Levine, S. In preparation. 2019.
"""
struct DiscreteFKL <: AbstractActorUpdate
    _temperature::Float32
end

function setup(
    up::DiscreteFKL,
    ::AbstractEnvironment,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    qf::AbstractValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{DiscreteFKL}
    return UpdateState(
        up,
        optim,
        (optim=Optimisers.setup(optim, π_θ), rng=rng),
    )
end

function update(
    st::UpdateState{DiscreteFKL},
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    states, # Must be >= 2D
)
    rng = Lux.replicate(st._state.rng)
    up = st._update

    # If the policy is discrete, we compute the gradient as equation (7)
    # in [1] by iterating over the actions. If using the hard DiscreteFKL then
    # equation (9) is used
    ∇π_θ = if up._temperature != 0
        # Soft DiscreteFKL
        q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
        q ./= up._temperature
        ℬq = softmax(q; dims=1)

        gradient(π_θ) do π_θ
            lnπ, π_st = logprob(π, π_f, π_θ, π_st, states)
            loss = -gpu_mean(sum(ℬq .* lnπ; dims=1))
            loss *= up._temperature
            loss
        end
    else
        # Hard DiscreteFKL
        q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
        maximal_actions = mapslices(x -> argmax_break_ties_randomly(rng, x), q; dims=1)
        maximal_actions = dropdims(maximal_actions; dims=1)

        gradient(π_θ) do π_θ
            lnπ, π_st = logprob(π, π_f, π_θ, π_st, states, maximal_actions)
            return -gpu_mean(lnπ)
        end
    end

    π_optim_state = st._state.optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    return UpdateState(
        st._update,
        st._optim,
        (
            optim = π_optim_state,
            rng = rng,
        ),
    ), π_θ, π_st, qf_st
end
