"""
    ContinuousRKL

ContinuousRKL implements a policy improvement operator which minimizes the reverse
KL-divergence between a learned policy and the Boltzmann distribution over action values
[1]. This operator is equivalent to the policy improvement operator used by SAC [2, 3] when
the constructor arguments are chosen appropriately. This update is more generally an
implementation of the RKL policy improvement operator in [1].

The constructor argument `reparameterised` denotes whether gradients are estimated using the
reparameterisation trick or not. `baseline_actions` determines how many actions to use in
estimating the state-value baselines, and `τ` is the temperature parameter. The
`num_samples` keyword determines how many actions are used to estimate the gradient, which
is typically `1`.

## Updates

This section discusses which update targets are used to update the actor and critic, as well
as some implementation details on how these updates are performed.

### Actor Update

For discrete actions, we would use the gradient in equation 6 in [1]:

    ∇RKL(π, ℬQ) =   𝔼_{π} [ τ⁻¹ Q(s, a) - ln(π(a | s))]                                 (1)
                =   -Σₐ ∇π(a | s) [ τ⁻¹ Q(s, a) - ln(π(a | s))]                         (2)
where:
    ∇ = ∇_ϕ
    π = π_ϕ
    τ = temperature argument

For continuous actions, we can no longer sum over all actions, instead, we
use a sampled (estimated) gradient as outlined in equation (10) of [1]:

    ∇RKL(π, ℬQ) =   𝔼_{π} [ [ (Q(s, a) - V(s))(τ⁻¹) - ln(π(a∣s)) ] ∇ln(π(a∣s)) ]        (3)
                ≈   -1/n  ∑₁ⁿ [ (Q(s, aᵢ) - V(s))(τ⁻¹) - ln(π(aᵢ∣s)) ] ∇ln(π(aᵢ∣s))     (4)
                ≈   ∇𝔼_{ε} [ (Q(s, f(ε)) - V(s))(τ⁻¹) - ln(π(f(ε)∣s)) ]                 (5)
                            where f(ε) = a
where:
    ∇ = ∇_ϕ
    π = π_ϕ
    τ = temperature argument

The gradient (3) is approximated by (4) or (5) using repeated random sampling,
usually using a single action sample aᵢ. To estimate this gradient, we can use two different
methods. (4) uses the likelihood trick while (5) uses the reparameterization trick.

In the continuous action setting, a state value baseline (V in equations (3-5) above) can be
used. The baseline is approximated as V(s) = 𝔼[q(s, A)] ≈ 1/n ∑ⁿᵢ₌₁ q(s, aᵢ).

In the discrete action setting, no baseline is required because we can compute the
expectation in equations (1) and (2) (equation (6) in [1]) exactly (assuming the true
value function is known).

### Implementation Details

This implementation of the RKL actor-critic algorithm allows for four different forms of the
RKL update. These are controlled by the fields `_scale_actor_lr_by_temperature` and
`_temperature_scales_entropy`. When `_temperature_scales_entropy`, then the temperature is
treated as an entropy scale as in [1, 4, 5]. When `not _temperature_scales_entropy`, then
the temperature is treated as a reward scale as in [2, 3]. The updates are as follows:

1. `scale_actor_lr_by_temperature and temperature_scales_entropy`:
    - Actor gradient = ∇𝔼[τ ln(π) - 𝔸]

2. `not scale_actor_lr_by_temperature and temperature_scales_entropy`:
    - Actor gradient = ∇𝔼[ln(π) - 𝔸/τ]

3. `scale_actor_lr_by_temperature and not temperature_scales_entropy`:
    - Actor gradient = ∇τ𝔼[ln(π) - 𝔸]

4. `not scale_actor_lr_by_temperature and not temperature_scales_entropy`:
    - Actor gradient = ∇𝔼[ln(π) - 𝔸]

where 𝔸 is the advantage function. All these updates are sensible but are implementation
details that can affect the algorithm's performance. The original SAC update in [2, 3] is
equivalent to (4), and this is the implementation used in the original SAC codebase.
The StableBaselines3 codebase [4] and SpinningUp codebase [5] use the implementation in (1).

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

[4] Stable-Baselines3: Reliable Reinforcement Learning Implementations.
Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann,
N. Journal of Machine Learning Research. 2021.

[5] Spinning Up in Deep Reinforcement Learning. Achiam, J. 2018.
"""
struct ContinuousRKL <: AbstractActorUpdate
    _reparam::Bool

    _baseline_actions::Int
    _temperature::Float32
    _num_samples::Int

    function ContinuousRKL(
        reparameterised::Bool, baseline_actions::Int, τ::Real, num_samples,
    )
        return new(reparameterised, baseline_actions, τ, num_samples)
    end
end

function ContinuousRKL(τ; reparam, baseline_actions, num_samples)
    return ContinuousRKL(reparam, baseline_actions, τ, num_samples)
end

function setup(
    up::ContinuousRKL,
    ::AbstractEnvironment,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{ContinuousRKL}
    @assert up._baseline_actions >= 0 "expected baseline_actions >= 0"
    @assert up._num_samples > 0 "expected num_samples > 0"
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
    st::UpdateState{ContinuousRKL},
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    states::AbstractArray, # Must be >= 2D
)
    up = st._update
    rng = Lux.replicate(st._state.rng)
    ∇π_θ, π_st, qf_st = if !up._reparam
        ll_grad(
            up, rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st,states,
        )
    else
        reparam_grad(
            up, rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states,
        )
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

function reparam_grad(
    up::ContinuousRKL,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    state_batch::AbstractArray, # Must be >= 2D
)
    batch_size = size(state_batch)[end]
    state_size = size(state_batch)[begin:end-1]

    # Estimate the approximate state value for each state using v(s) = 𝔼[q(s, A)]
    if up._baseline_actions > 0
        v, π_st, qf_st = _get_baseline(
            rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, state_batch,
            up._baseline_actions,
        )
    end

    ∇π_θ = gradient(π_θ) do θ
        actions, lnπ, π_st = rsample_with_logprob(
            π, rng, π_f, θ, π_st, state_batch; num_samples=up._num_samples,
        )
        action_size = size(actions, 1)
        actions = reshape(actions, action_size, :)

        states = repeat(
            state_batch;
            inner = (ones(Int, length(state_size))..., up._num_samples),
        )

        # Compute the advantage
        q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states, actions)
        q = reshape(q, up._num_samples, batch_size)
        adv = up._baseline_actions > 0 ? q .- v : q

        lnπ = reshape(lnπ, size(adv))

        # Compute the reparameterized loss
        ∇π = if up._temperature != 0
            # Soft ContinuousRKL
            @. adv - up._temperature * lnπ
        else
            # Hard ContinuousRKL
            adv
        end
        -gpu_mean(∇π)
    end

    return ∇π_θ, π_st, qf_st
end

function ll_grad(
    up::ContinuousRKL,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::Q,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    state_batch::AbstractArray, # Must be >= 2D
)
    batch_size = size(state_batch)[end]
    state_size = size(state_batch)[begin:end-1]

    # Estimate the approximate state value for each state using v(s) = 𝔼[q(s, A)]
    if up._baseline_actions > 0
        v, π_st, qf_st = _get_baseline(
            rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, state_batch,
            up._baseline_actions,
        )
    end

    ∇π = gradient(π_θ) do θ
        # Sample actions for each state in the batch, needed to estimate the gradient in
        # each state: ∇J = 𝔼[q(S, A) - τ*ln(π(A∣S)]
        actions, lnπ, π_st = sample_with_logprob(
            π, rng, π_f, θ, π_st, state_batch; num_samples=up._num_samples,
        )
        action_size = size(actions, 1)
        actions = reshape(actions, :, batch_size)

        # Repeat states for each action sample in each state
        states = repeat(
            state_batch;
            inner = (ones(Int, length(state_size))..., up._num_samples),
        )

        # Calculate the advantage
        q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states, actions; reduct=true)
        q = reshape(q, up._num_samples, batch_size)
        adv = up._baseline_actions > 0 ? q .- v : q

        lnπ = reshape(lnπ, size(adv))

        scale = if up._temperature != 0
            # Soft ContinuousRKL
            @. adv - up._temperature * lnπ
        else
            # Hard ContinuousRKL
            adv
        end
        scale = ChainRulesCore.ignore_derivatives(scale)

        -gpu_mean(lnπ .* scale)
    end

    return ∇π, π_st, qf_st
end

function _get_baseline(
    rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, state_batch, num_samples,
)::Tuple{AbstractArray{Float32},<:NamedTuple,<:NamedTuple}
    if num_samples > 0
        batch_size = size(state_batch)[end]
        state_size = size(state_batch)[begin:end-1]

        actions, π_st = sample(
            π, rng, π_f, π_θ, π_st, state_batch; num_samples=num_samples,
        )
        action_size = size(actions, 1)
        actions = reshape(actions, action_size..., :)

        states = repeat(
            state_batch;
            inner = (ones(Int, length(state_size))..., num_samples),
        )

        q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states, actions; reduct=true)
        q = reshape(q, num_samples, batch_size)
        return ChainRulesCore.ignore_derivatives(mean(q; dims=1)), π_st, qf_st
    else
        error("cannot compute baseline with 0 actions")
    end
end

