struct ContinuousProximalMDCCEM <: AbstractActorUpdate
    _n::Int
    _ρ::Float32
    _ρ̃::Float32

    # Entropy regularization
    _τ::Float32         # Actor Temperature
    _τ̃::Float32         # Proposal Temperature

    # Number of samples for the KL/entropy term
    _π_num_samples::Int
    _π̃_num_samples::Int

    # Clipping ratio for RKL
    _clip::Float32

    _inv_λ::Float32
    _inv_λ̃::Float32
    _num_md_updates::Int

    _forward_direction::Bool

    function ContinuousProximalMDCCEM(
        n::Int, ρ::Real, ρ̃::Real, τ::Real, π_num_samples::Int, τ̃::Real,
        π̃_num_samples::Int, λ::Real, λ̃::Real, num_md_updates::Integer,
        forward_direction, clip::Real,
    )
        @assert (τ >= 0) "expected τ >= 0"
        @assert (τ̃ >= 0) "expected τ̃ >= 0"
        @assert (π_num_samples >= 0) "expected π_num_samples >= 0"
        @assert (π̃_num_samples >= 0) "expected π̃_num_samples >= 0"
        @assert (ρ >= 0) "expected ρ >= 0"
        @assert (ρ̃ >= 0) "expected ρ̃ >= 0"
        @assert (trunc(n * ρ) > 0) "expected ⌊ρn⌋ > 0"
        @assert (trunc(n * ρ̃) > 0) "expected ⌊ρ̃n⌋ > 0"

        return new(
            n, ρ, ρ̃, τ, τ̃, π_num_samples, π̃_num_samples, clip, inv(λ), inv(λ̃),
            num_md_updates, forward_direction,
        )
    end
end

function ContinuousProximalMDCCEM(
    n::Int, ρ::Real, ρ̃::Real, τ::Real, num_entropy_samples::Int, λ::Real,
    num_md_updates::Integer, forward_direction, clip::Real,
)
    return ContinuousProximalMDCCEM(
        n, ρ, ρ̃, τ, num_entropy_samples, τ, num_entropy_samples, λ, λ, num_md_updates,
        forward_direction, clip,
    )
end

function setup(
    up::ContinuousProximalMDCCEM,
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
)::UpdateState{ContinuousProximalMDCCEM}
    π̃ = deepcopy(π)
    π̃_f = deepcopy(π_f)
    π̃_θ = deepcopy(π_θ)
    π̃_st = deepcopy(π_st)
    π̃_optim = deepcopy(optim)

    return setup(
        up, π, π_f, π_θ, π_st, π̃, π̃_f, π̃_θ, π̃_st, qf, qf_f, qf_θ, qf_st, optim, π̃_optim,
        rng,
    )
end

function setup(
    up::ContinuousProximalMDCCEM,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    π̃::AbstractContinuousParameterisedPolicy,
    π̃_f,    # proposal policy model
    π̃_θ,    # proposal policy model parameters
    π̃_st,   # proposal policy model state
    qf::AbstractActionValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    π_optim::Optimisers.AbstractRule,
    π̃_optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{ContinuousProximalMDCCEM}
    return UpdateState(
        up,
        π_optim,
        (
            π_optim = Optimisers.setup(π_optim, π_θ),
            π̃_optim = Optimisers.setup(π̃_optim, π̃_θ),
            π̃ = π̃,
            π̃_f = π̃_f,
            π̃_θ = π̃_θ,
            π̃_st = π̃_st,
            rng = Lux.replicate(rng),
            # Previous policy parameters for the KL update
            θ_t = π_θ,    # These are immutable
            π_state_t = π_st,  # These are immutable
            θ̃_t = π̃_θ,    # These are immutable
            π̃_state_t = π̃_st,  # These are immutable
            current_update = 1,
        )
    )
end

function update(
    st::UpdateState{ContinuousProximalMDCCEM},
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

    # Unpack proposal policy from update state
    π̃ = st._state.π̃
    π̃_f = st._state.π̃_f
    π̃_θ = st._state.π̃_θ
    π̃_st = st._state.π̃_st

    # Unpack previous policy parameters
    θ_t = st._state.θ_t
    π_state_t = st._state.π_state_t
    θ̃_t = st._state.θ̃_t
    π̃_state_t = st._state.π̃_state_t

    # Get the actions of maximal value as ordered by the action-value function critic
    π_top_actions, π̃_top_actions, π̃_st = _get_top_actions(
        rng, states, π̃, π̃_f, π̃_θ, π̃_st, qf, qf_f, qf_θ, qf_st, up._n, up._ρ, up._ρ̃
    )

    # Compute the actor and proposal gradients using the actions of maximal value gotten
    # above
    ∇π_θ, π_st, π_state_t = if up._forward_direction
        _fkl_gradient(
            up, rng, π, π_f, π_θ, π_st, θ_t, π_state_t, up._τ, up._π_num_samples, up._inv_λ,
            states, π_top_actions,
        )
    else
        _rkl_gradient(
            up, rng, π, π_f, π_θ, π_st, θ_t, π_state_t, up._τ, up._π_num_samples, up._inv_λ,
            states, π_top_actions,
        )
    end

    ∇π̃_θ, π̃_st, π̃_state_t = if up._forward_direction
        _fkl_gradient(
            up, rng, π̃, π̃_f, π̃_θ, π̃_st, θ̃_t, π̃_state_t, up._τ̃, up._π̃_num_samples,
            up._inv_λ̃, states, π̃_top_actions
        )
    else
        _rkl_gradient(
            up, rng, π̃, π̃_f, π̃_θ, π̃_st, θ̃_t, π̃_state_t, up._τ̃, up._π̃_num_samples,
            up._inv_λ̃, states, π̃_top_actions
        )
    end

    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))
    π̃_optim_state = st._state.π̃_optim
    π̃_optim_state, π̃_θ = Optimisers.update(π̃_optim_state, π̃_θ, only(∇π̃_θ))

    next_update = mod(st._state.current_update, up._num_md_updates) + 1
    return UpdateState(
        st._update,
        st._optim,
        (
            π_optim = π_optim_state,
            π̃_optim = π̃_optim_state,
            π̃ = π̃,
            π̃_f = π̃_f,
            π̃_θ = π̃_θ,
            π̃_st = π̃_st,
            rng = rng,
            θ_t = next_update == 1 ? π_θ : θ_t,
            π_state_t = next_update == 1 ? π_st : π_state_t,
            θ̃_t = next_update == 1 ? π_θ : θ̃_t,
            π̃_state_t = next_update == 1 ? π̃_st : π̃_state_t,
            current_update = next_update,
        )
    ), π_θ, π_st, qf_st
end

function _rkl_gradient(
    up::ContinuousProximalMDCCEM,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    θ_t,        # Policy parameters at timestep t
    state_t,    # Policy model state at timestep t
    τ,
    num_samples,
    inv_λ,
    states::AbstractArray,
    top_actions,
)
    params_t, state_t = get_params(π, π_f, θ_t, state_t, states)
    lnπ_t_top = logprob(π, top_actions, params_t)

    ∇π_θ = gradient(π_θ) do θ
        params, π_st = get_params(π, π_f, θ, π_st, states)
        lnπ_θ_top = logprob(π, top_actions, params)

        lr_term = ChainRulesCore.ignore_derivatives(
            exp.(lnπ_θ_top .- lnπ_t_top)
        )
        lr_term = if !isinf(up._clip)
            clamp.(lr_term, inv(1 + up._clip), 1 + up._clip) .* lnπ_θ_top
        else
            lr_term .* lnπ_θ_top
        end

        samples, lnπ_θ, π_st = sample_with_logprob(
            π, rng, π_f, θ, π_st, states; num_samples,
        )
        lnπ_t = logprob(π, samples, params_t)

        scale = ChainRulesCore.ignore_derivatives(
            ((τ - inv_λ) .* lnπ_t) .+ (up._inv_λ .* lnπ_θ)
        )
        kl_entropy_term = scale .* lnπ_θ

        gpu_mean(kl_entropy_term) - gpu_mean(lr_term)
    end

    return ∇π_θ, π_st, state_t
end

function _fkl_gradient(
    up::ContinuousProximalMDCCEM,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    θ_t,        # Policy parameters at timestep t
    state_t,    # Policy model state at timestep t
    τ,
    num_samples,
    inv_λ,
    states::AbstractArray,
    top_actions,
)
    params_t, state_t = get_params(π, π_f, θ_t, state_t, states)

    n_entropy_samples = num_samples
    samples_π_t, lnπ_t, state_t = sample_with_logprob(
        π, rng, π_f, θ_t, state_t, states; num_samples=num_samples + n_entropy_samples,
    )
    entropy_π_t = -mean(lnπ_t[end-n_entropy_samples+1:end, :]; dims=1)
    lnπ_t = lnπ_t[begin:begin+num_samples-1, :]
    ς = 1 .+ up._τ .* (lnπ_t .+ entropy_π_t)
    samples = samples_π_t[:, begin:begin+num_samples-1, :]

    ∇π_θ = gradient(π_θ) do θ
        params, π_st = get_params(π, π_f, θ, π_st, states)

        best_lnπ_θ = logprob(π, top_actions, params)
        lnπ_θ = logprob(π, samples, params)

        kl_entropy_term = (ς .- up._inv_λ) .* lnπ_θ

        gpu_mean(kl_entropy_term) - gpu_mean(best_lnπ_θ)
    end

    return ∇π_θ, π_st, state_t
end
