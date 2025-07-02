struct ContinuousMPO <: AbstractActorUpdate
    _n::Int
    _kl_policy_coeff::Float32

    _baseline_actions::Int

    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature
    _num_entropy_samples::Int

    function ContinuousMPO(
        n::Int, kl_policy_coeff::Real, baseline_actions::Int, τ::Real,
        num_entropy_samples::Int,
    )
        @assert (τ >= 0) "expected τ >= 0"
        @assert (kl_policy_coeff > 0) "expected kl_policy_coeff > 0"
        @assert baseline_actions >= 0 "expected baseline_actions >= 0"
        @assert n > 0 "expected n > 0"
        @assert (num_entropy_samples >= 0) "expected num_entropy_samples >= 0"

        return new(n, kl_policy_coeff, baseline_actions, τ, num_entropy_samples)
    end
end

function ContinuousMPO(
    kl_policy_coeff::Real, τ::Real;
    baseline_actions::Int=0, n::Int=1, num_entropy_samples::Int=1,
)
    return ContinuousMPO(n, kl_policy_coeff, baseline_actions, τ, num_entropy_samples)
end

function setup(
    up::ContinuousMPO,
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
)::UpdateState{ContinuousMPO}
    return UpdateState(
        up,
        optim,
        (π_optim = Optimisers.setup(optim, π_θ), rng = Lux.replicate(rng))
    )
end

function update(
    st::UpdateState{ContinuousMPO},
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    qf::Q,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    up = st._update
    rng = Lux.replicate(st._state.rng)

    # Get the gradient
    ∇π_θ, π_st, qf_st = _gradient(
        up, rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states,
    )

    # Update
    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    return UpdateState(
        st._update,
        st._optim,
        (π_optim = π_optim_state, rng = rng,),
    ), π_θ, π_st, qf_st
end

function _gradient(
    up::ContinuousMPO,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    qf::Q,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    # Sample actions for entropy regularization and the policy update. Use a pooled batch of
    # samples for both (i.e. the same actions may be used for the entropy and KL(π_{KL} ||
    # π_{θ}) calculations)
    n_samples = max(up._n, up._num_entropy_samples)
    actions, π_st = sample(
        π, rng, π_f, π_θ, π_st, states; num_samples=n_samples,
    )
    if up._τ > 0 && up._num_entropy_samples > 0
        entropy_samples = actions[:, begin:up._num_entropy_samples, :]
    end
    actions = actions[:, begin:up._n, :]

    action_size = size(actions, 1)
    reshaped_actions = reshape(actions, action_size..., :)

    # Stack states to calculate action values. Each sampled action requires one state
    # observation
    state_size = size(states)[begin:end-1]
    stacked_states = repeat(
        states;
        inner = (ones(Int, length(state_size))..., up._n),
    )

    # Compute advantage function, needed for the π_KL importance sampling ratio
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, stacked_states, reshaped_actions)
    batch_size = size(states)[end]
    q = reshape(q, up._n, batch_size)

    batch_size = size(states)[end]
    adv = if up._baseline_actions > 0
        v, π_st, qf_st = _get_baseline(
            rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states,
            up._baseline_actions,
        )
        q .- v
    else
        q
    end

    # Construct the importance sampling ratio scale. For numerical stability, subtract the
    # maximum before exponentiating.
    scale = adv ./ up._kl_policy_coeff
    scale .-= (size(scale, 1) > 1 ? maximum(scale; dims=1) : maximum(scale))
    scale .= exp.(scale)

    ∇π_θ = gradient(π_θ) do θ
        params, π_st = get_params(π, π_f, θ, π_st, states)
        lnπ = logprob(π, actions, params)

        if up._τ > 0 && up._num_entropy_samples > 0
            entropy_lnπ = logprob(π, entropy_samples, params)
            entropy = -(entropy_lnπ .^ 2) ./ 2

            -gpu_mean(scale .* lnπ) - up._τ * gpu_mean(entropy)
        else
            -gpu_mean(scale .* lnπ)
        end
    end

    return ∇π_θ, π_st, qf_st
end
