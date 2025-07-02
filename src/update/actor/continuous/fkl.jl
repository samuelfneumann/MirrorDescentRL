"""
    ContinuousFKL

ContinuousFKL implements a policy improvement operator which minimizes the forward
KL-divergence between a learned policy and the Boltzmann distribution over action values
[1]. This update is an implementation of the FKL policy improvement operator in [1].

## Updates

Because this implementation solely follows [1], the temperature τ is the entropy scale,
rather than a reward scale as in [2, 3].

### Actor Update

For continuous actions, we use a sampled gradient estimate as outlined in equation (14) of
[1]:

    ∇FKL(π, ℬQ) = - ∑₁ⁿ ρ̃ ∇ln(π(a∣s))                                                   (3)

where:
    ρ_i = ℬQ(a_i | s) / π_θ(a_i | s) ∝ exp(Q(s, a_i)τ⁻¹) / π(a_i | s)
    ρ̃_i = ρ_i / ∑(ρ_j)
    ∇ = ∇_ϕ
    π = π_ϕ

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
struct ContinuousFKL <: AbstractActorUpdate
    _temperature::Float32
    _num_samples::Int

    function ContinuousFKL(τ::Real; num_samples)
        @assert num_samples > 1
        return new(τ, num_samples, true)
    end
end

function setup(
    up::ContinuousFKL,
    π::AbstractParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    qf::AbstractValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{ContinuousFKL}
    return UpdateState(
        up,
        optim,
        (
            optim = Optimisers.setup(optim, π_θ),
            rng = Lux.replicate(rng),
        ),
    )
end

function update(
    st::UpdateState{ContinuousFKL},
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    states, # Must be >= 2D
)
    up = st._update
    rng = Lux.replicate(st._state.rng)

    batch_size = size(states)[end]
    state_size = size(states)[begin:end-1]

    # Sample actions for each state in the batch, needed to estimate the gradient in each
    # state
    actions, lnπ, π_st = sample_with_logprob(
        π, rng, π_f, π_θ, π_st, states; num_samples=up._num_samples,
    )
    action_size = size(actions, 1)
    actions = reshape(actions, action_size..., :)

    states = repeat(
        states;
        inner = (ones(Int, length(state_size))..., up._num_samples),
    )

    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states, actions)
    q = reshape(q, up._num_samples, batch_size)
    # lnprob, π_st = logprob(π, π_f, π_θ, π_st, states, actions)
    prob = exp.(lnπ)
    prob = reshape(prob, up._num_samples, batch_size)

    # Calculate the weighted importance sampling ratio
    # Right now, we follow equation (14) in [1] to compute the
    # weighted importance sampling ratio, where:
    #
    # ρ_i = BQ(a_i | s) / π_θ(a_i | s) ∝ exp(Q(s, a_i)τ⁻¹) / π(a_i | s)
    # ρ̂_i = ρ_i / ∑(ρ_j)
    #
    # We could compute a more numerically stable weighted importance
    # sampling ratio if needed (but the implementation is very
    # complicated):
    #
    # ρ̂ = π(a_i | s) [∑_{i≠j} ([h(s, a_j)/h(s, a_i)] * π(a_j | s)⁻¹) + 1]
    # h(s, a_j, a_i) = exp[(Q(s, a_j) - M)τ⁻¹] / exp[(Q(s, a_i) - M)τ⁻¹]
    # M = M(a_j, a_i) = max(Q(s, a_j), Q(s, a_i))
    weighted_lr = let
        # Honestly, I have no idea where I got these extra implementation details from. They
        # aren't found in [1] as far as I can tell, but they seem to work well. The details
        # I am referring to are marked by (*)
        likelihood_ratio = q ./ up._temperature
        max_lr = maximum(likelihood_ratio; dims=1)                      # (*)
        likelihood_ratio .-= max_lr                                     # (*)
        likelihood_ratio = exp.(likelihood_ratio)

        z = sum(likelihood_ratio; dims=1)                               # (*)
        likelihood_ratio ./= z                                          # (*)
        likelihood_ratio ./= prob

        weight = sum(likelihood_ratio; dims=1)

        likelihood_ratio ./ weight
    end

    # Compute the gradient of the ContinuousFKL objective
    ∇π_θ = gradient(π_θ) do π_θ
        logπ, π_st = logprob(π, π_f, π_θ, π_st, states, actions)
        logπ = reshape(logπ, up._num_samples, batch_size)

        loss = weighted_lr .* logπ
        loss = -gpu_mean(sum(loss; dims=1))

        # This is equivalent to scaling the actor stepsize by the temperature. Since we do
        # this in the RKL implementation, we also should do this in the FKL implementation
        # here.
        loss *= up._temperature

        return loss
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

