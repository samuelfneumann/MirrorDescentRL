struct ContinuousProximalMDMPO <: AbstractActorUpdate
    _n::Int
    _kl_policy_coeff::Float32

    _baseline_actions::Int

    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature

    _num_md_updates::Int
    _inv_λ::Float32

    # Whether the KL penalty is in the forward or reverse direction
    _forward_direction::Bool

    function ContinuousProximalMDMPO(
        n::Integer, kl_policy_coeff::Real, baseline_actions::Integer, τ::Real,
        md_λ::Real, num_md_updates::Integer, forward_direction::Bool,
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"
        @assert (kl_policy_coeff > 0) "expected kl_policy_coeff > 0"
        @assert baseline_actions >= 0 "expected baseline_actions >= 0"
        @assert n > 0 "expected n > 0"

        return new(
            n, kl_policy_coeff, baseline_actions, τ, num_md_updates, inv(md_λ),
            forward_direction,
        )
    end
end

function ContinuousProximalMDMPO(
    kl_policy_coeff::Real, τ::Real, λ::Real, num_md_updates::Int;
    baseline_actions::Int=0, n::Int=1, forward_direction::Bool,
)
    ContinuousProximalMDMPO(
        n, kl_policy_coeff, baseline_actions, τ, λ, num_md_updates, forward_direction,
    )
end

function setup(
    up::ContinuousProximalMDMPO,
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
)::UpdateState{ContinuousProximalMDMPO}
    return UpdateState(
        up,
        optim,
        (
            π_optim = Optimisers.setup(optim, π_θ),
            rng = Lux.replicate(rng),
            # Previous policy parameters for the KL update
            θ_t = π_θ,       # These are immutable
            state_t = π_st,  # These are immutable
            current_update = 1,
        )
    )
end

function update(
    st::UpdateState{ContinuousProximalMDMPO},
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

    # Frozen current policy parameters, must stay fixed during the MD update and only update
    # every up._num_md_updates
    θ_t = st._state.θ_t
    # State of the current policy, which will change during the MD update
    state_t = st._state.state_t

    # Get the gradient
    ∇π_θ, π_st, qf_st, state_t = if up._forward_direction
        _fkl_gradient(
            up, rng, π, π_f, π_θ, π_st, θ_t, state_t, qf, qf_f, qf_θ, qf_st, states,
        )
    else
        _rkl_gradient(
            up, rng, π, π_f, π_θ, π_st, θ_t, state_t, qf, qf_f, qf_θ, qf_st, states,
        )
    end

    # Update
    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    next_update = mod(st._state.current_update, up._num_md_updates) + 1
    return UpdateState(
        st._update,
        st._optim,
        (
            π_optim = π_optim_state,
            rng = rng,
            θ_t = next_update == 1 ? π_θ : θ_t,
            state_t = next_update == 1 ? π_st : state_t,
            current_update = next_update,
        ),
    ), π_θ, π_st, qf_st
end

function _rkl_gradient(
    up::ContinuousProximalMDMPO,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    θ_t,
    state_t,
    qf::Q,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    # Sample actions for entropy regularization and the policy update. Use a pooled batch of
    # samples for both (i.e. the same actions may be used for the entropy and KL(π_{KL} ||
    # π_{θ}) calculations)
    actions, π_st = sample(π, rng, π_f, π_θ, π_st, states; num_samples=up._n)

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

    params_t, state_t = get_params(π, π_f, θ_t, state_t, states)
    lnπ_t = logprob(π, actions, params_t)

    scale = adv ./ up._kl_policy_coeff
    scale .-= maximum(scale)
    scale .= exp.(scale)

    ∇π_θ = gradient(π_θ) do θ
        params, π_st = get_params(π, π_f, θ, π_st, states)
        lnπ_θ = logprob(π, actions, params)

        scale = ChainRulesCore.ignore_derivatives(
            scale - ((up._τ - up._inv_λ) .* lnπ_t) - (up._inv_λ .* lnπ_θ)
        )
        -gpu_mean(scale .* lnπ_θ)
    end

    return ∇π_θ, π_st, qf_st, state_t
end

function _fkl_gradient(
    up::ContinuousProximalMDMPO,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    θ_t,
    state_t,
    qf::Q,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    n_entropy_samples = up._n
    actions, lnπ_t, state_t = sample_with_logprob(
        π, rng, π_f, θ_t, state_t, states; num_samples=(up._n + n_entropy_samples),
    )
    entropy_π_t = -mean(lnπ_t[end-n_entropy_samples+1:end, :]; dims=1)
    lnπ_t = lnπ_t[begin:begin+up._n-1, :]
    actions = actions[:, begin:begin+up._n-1, :]

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

    π_KL_scale = adv ./ up._kl_policy_coeff
    π_KL_scale .-= maximum(π_KL_scale)
    π_KL_scale .= exp.(π_KL_scale)

    ς = 1 .+ up._τ .* (entropy_π_t .+ lnπ_t)

    ∇π_θ = gradient(π_θ) do θ
        params, π_st = get_params(π, π_f, θ, π_st, states)
        lnπ_θ = logprob(π, actions, params)

        scale = ChainRulesCore.ignore_derivatives(
            π_KL_scale .- ς .+ up._inv_λ
        )
        -gpu_mean(scale .* lnπ_θ)
    end

    return ∇π_θ, π_st, qf_st, state_t
end

# function _reparam_rkl_gradient(
#     up::ContinuousProximalMDMPO,
#     rng::AbstractRNG,
#     π::AbstractContinuousParameterisedPolicy,
#     π_f,    # actor policy model
#     π_θ,    # actor policy model parameters
#     π_st,   # actor policy model state
#     θ_t,
#     state_t,
#     qf::Q,
#     qf_f,
#     qf_θ,
#     qf_st,
#     states::AbstractArray, # Must be >= 2D
# )
#     @warn "can't get this function to work..."
#     batch_size = size(states)[end]
#     baseline = if up._baseline_actions > 0
#         v, π_st, qf_st = _get_baseline(
#             rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states,
#             up._baseline_actions,
#         )
#         v
#     else
#         0
#     end

#     # Stack states to calculate action values. Each sampled action requires one state
#     # observation
#     state_size = size(states)[begin:end-1]
#     stacked_states = repeat(
#         states;
#         inner = (ones(Int, length(state_size))..., up._n),
#     )

#     params_t, state_t = get_params(π, π_f, θ_t, state_t, states)

#     ∇π_θ = gradient(π_θ) do θ
#         # Sample actions for entropy regularization and the policy update. Use a pooled batch of
#         # samples for both (i.e. the same actions may be used for the entropy and KL(π_{KL} ||
#         # π_{θ}) calculations)
#         actions, π_st = rsample(π, rng, π_f, θ, π_st, states; num_samples=up._n)

#         action_size = size(actions, 1)
#         reshaped_actions = reshape(actions, action_size..., :)

#         # Compute advantage function, needed for the π_KL importance sampling ratio
#         q, qf_st = predict(qf, qf_f, qf_θ, qf_st, stacked_states, reshaped_actions)
#         q = reshape(q, up._n, batch_size)
#         adv = baseline != 0 ? q .- baseline : q

#         scale = adv ./ up._kl_policy_coeff
#         scale = scale .- maximum(scale)
#         scale = exp.(scale)

#         params, π_st = get_params(π, π_f, θ, π_st, states)
#         lnπ_θ = logprob(π, actions, params)
#         lnπ_t = logprob(π, actions, params_t)

#         loss = scale - ((up._τ - up._inv_λ) .* lnπ_t) - (up._inv_λ .* lnπ_θ)
#         -gpu_mean(loss)
#     end

#     return ∇π_θ, π_st, qf_st, state_t
# end


