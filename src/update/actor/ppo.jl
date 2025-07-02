struct PPO <: AbstractPolicyGradientStyleUpdate
    _clip_ratio::Float32
    _n_updates::Int
    _spread_across_env_steps::Bool
    _entropy_scale::Float32
end

function setup(
    up::PPO,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)
    return UpdateState(
        up,
        optim,
        (
            π_optim = Optimisers.setup(optim, π_θ),
            π_θ_old = π_θ,
            π_st_old = π_st,
            rng = rng,
            current_update = 1,
        ),
    )
end

function update(
    st::UpdateState{PPO},
    π::AbstractParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    s_t::AbstractArray, # Must be >= 2D
    a_t::AbstractArray,
    A_t::AbstractVector,
    γ_t::AbstractVector,
)
    up = st._update
    rng = Lux.replicate(st._state.rng)
    π_θ_old = st._state.π_θ_old
    π_st_old = st._state.π_st_old

    lnπ_old, _ = logprob(π, π_f, π_θ_old, π_st_old, s_t, a_t)
    π_optim_state = st._state.π_optim
    n_updates = up._spread_across_env_steps ? 1 : up._n_updates
    for _ in 1:n_updates
        ∇π_θ = gradient(π_θ) do θ
            lnπ, π_st, = logprob(π, π_f, θ, π_st, s_t, a_t)
            ratio = exp.(lnπ - lnπ_old)
            clipped_adv = clamp.(
                ratio,
                one(up._clip_ratio) - up._clip_ratio,
                one(up._clip_ratio) + up._clip_ratio,
            ) .* A_t
            scaled_adv = ratio .* A_t
            loss = min.(scaled_adv, clipped_adv)

            entropy, π_st = _calculate_entropy(up, π, π_f, θ, π_st, s_t, rng)

            -mean(loss) - up._entropy_scale .* mean(entropy)
        end

        # @data exp grad_norm=norm(only(∇π_θ) |> ComponentArray)
        π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))
    end

    _log_ppo_data(π, π_f, π_θ, π_st, s_t)

    next_update = mod(st._state.current_update, up._n_updates) + 1
    π_θ_old, π_st_old = if (
        !up._spread_across_env_steps ||
        up._spread_across_env_steps && next_update == 1
    )
        π_θ, π_st
    else
        π_θ_old, π_st_old
    end

    return UpdateState(
        st._update,
        st._optim,
        (
            π_optim = π_optim_state,
            π_θ_old = π_θ_old,
            π_st_old = π_st_old,
            rng = rng,
            current_update = next_update,
        ),
    ), π_θ, π_st
end

function _calculate_entropy(
    ::PPO, π::AbstractContinuousParameterisedPolicy, π_f, π_θ, π_st, s_t, rng,
)
    _, lnπ, π_st = sample_with_logprob(π, rng, π_f, π_θ, π_st, s_t)
    entropy = - (lnπ .^ 2) ./ 2
    return entropy, π_st
end

function _calculate_entropy(
    ::PPO, π::AbstractDiscreteParameterisedPolicy, π_f, π_θ, π_st, s_t, rng,
)
    return entropy(π, π_f, π_θ, π_st, s_t)
end

function _log_ppo_data(π::AbstractContinuousParameterisedPolicy, π_f, π_θ, π_st, s_t)
    params = get_params(π, π_f, π_θ, π_st, s_t)[1]

    policy = π(params...)
    _entropy_est = discrete_entropy.(policy)
    _entropy_est = mean(_entropy_est; dims=2)

    @data exp entropy_estimate=_entropy_est
end

function _log_ppo_data(π::AbstractDiscreteParameterisedPolicy, π_f, π_θ, π_st, s_t)
    entropy, π_st = Distributions.entropy(π, π_f, π_θ, π_st, s_t)
    @data exp entropy_estimate=mean(entropy)
end
