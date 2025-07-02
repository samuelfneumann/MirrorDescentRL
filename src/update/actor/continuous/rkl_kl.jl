"""
    ContinuousRKLKL

ContinuousRKLKL implements the RKL policy update with a KL penalty
"""
struct ContinuousRKLKL <: AbstractActorUpdate
    _reparam::Bool

    _baseline_actions::Int
    _temperature::Float32
    _num_samples::Int

    _inv_λ::Float32      # Inverse stepsize for mirror descent (functional) update
    # TODO: this should be renamed to num_kl_updates
    _num_md_updates::Int

    _forward_direction::Bool

    function ContinuousRKLKL(
        reparameterised::Bool, baseline_actions::Int, τ::Real, md_λ::Real,
        num_md_updates::Int, forward_direction::Bool, num_samples::Int,
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (num_samples >= 1) "expected num_samples >= 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"

        msg = "cannot use reparameterised gradient estimation when forward_direction=true"
        @assert !(forward_direction && reparameterised) msg

        return new(
            reparameterised, baseline_actions, τ, num_samples, inv(md_λ), num_md_updates,
            forward_direction,
        )
    end
end

function ContinuousRKLKL(
    τ, md_λ, num_md_updates;
    forward_direction, num_samples, baseline_actions, reparam
)
    ContinuousRKLKL(
        reparam, baseline_actions, τ, md_λ, num_md_updates, forward_direction, num_samples,
    )
end

function setup(
    up::ContinuousRKLKL,
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
    rng::AbstractRNG;
)::UpdateState{ContinuousRKLKL}
    return UpdateState(
        up,
        optim,
        (
            optim = Optimisers.setup(optim, π_θ),
            rng = Lux.replicate(rng),
            θ_t = π_θ,    # These are immutable
            state_t = π_st,  # These are immutable
            current_update = 1,
        ),
    )
end

function update(
    st::UpdateState{ContinuousRKLKL},
    π::AbstractContinuousParameterisedPolicy,
    π_f,                    # policy model
    π_θ,                    # policy model parameters
    π_st,                   # policy model state
    qf::Q,
    qf_f,                   # q function model
    qf_θ,                   # q function model parameters
    qf_st,                  # q function model state
    states::AbstractArray,  # Must be >= 2D
)
    up = st._update
    rng = Lux.replicate(st._state.rng)

    # Frozen current policy parameters, must stay fixed during the MD update and only update
    # every up._num_md_updates
    θ_t = st._state.θ_t
    # State of the current policy, which will change during the MD update
    state_t = st._state.state_t

    ∇π_θ, π_st, qf_st, state_t = if !up._reparam
        if up._forward_direction
            _fkl_ll_grad(
                up, rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states, θ_t, state_t,
            )
        else
            _rkl_ll_grad(
                up, rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states, θ_t, state_t,
            )
        end
    else
        if up._forward_direction
            error("cannot use reparameterised gradient estimation with an FKL penalty")
        else
            _rkl_reparam_grad(
                up, rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states, θ_t, state_t,
            )
        end
    end

    π_optim_state = st._state.optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    next_update = mod(st._state.current_update, up._num_md_updates) + 1
    return UpdateState(
        st._update,
        st._optim,
        (
            optim = π_optim_state,
            rng = rng,
            θ_t = next_update == 1 ? π_θ : θ_t,
            state_t = next_update == 1 ? π_st : state_t,
            current_update = next_update,
        ),
    ), π_θ, π_st, qf_st
end

function _rkl_ll_grad(
    up::ContinuousRKLKL,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,                            # policy model
    π_θ,                            # policy model parameters
    π_st,                           # policy model state
    qf::Q,
    qf_f,                           # q function model
    qf_θ,                           # q function model parameters
    qf_st,                          # q function model state
    state_batch::AbstractArray,     # Must be >= 2D
    θ_t,                            # Previous policy parameters,
    state_t,                           # Previous policy state
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

    ∇π = gradient(π_θ) do π_θ
        # Sample actions for each state in the batch, needed to estimate the gradient in
        # each state: ∇J = 𝔼[q(S, A) - τ*ln(π(A∣S)]
        actions, lnπ_θ, π_st = sample_with_logprob(
            π, rng, π_f, π_θ, π_st, state_batch; num_samples=up._num_samples,
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

        lnπ_θ = reshape(lnπ_θ, size(adv))
        lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, states, actions)
        lnπ_t = reshape(lnπ_t, size(adv))

        scale = ChainRulesCore.ignore_derivatives(
            -adv .- up._inv_λ .* lnπ_t .+ (up._temperature + up._inv_λ) .* lnπ_θ
        )

        gpu_mean(lnπ_θ .* scale)
    end

    return ∇π, π_st, qf_st, state_t
end

function _rkl_reparam_grad(
    up::ContinuousRKLKL,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,                            # policy model
    π_θ,                            # policy model parameters
    π_st,                           # policy model state
    qf::Q,
    qf_f,                           # q function model
    qf_θ,                           # q function model parameters
    qf_st,                          # q function model state
    state_batch::AbstractArray,     # Must be >= 2D
    θ_t,                            # Previous policy parameters,
    state_t,                           # Previous policy state
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

    ∇π_θ = gradient(π_θ) do π_θ
        actions, lnπ_θ, π_st = rsample_with_logprob(
            π, rng, π_f, π_θ, π_st, state_batch; num_samples=up._num_samples,
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

        lnπ_θ = reshape(lnπ_θ, size(adv))
        lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, states, actions)
        lnπ_t = reshape(lnπ_t, size(adv))

        loss = -adv .- up._inv_λ .* lnπ_t .+ (up._temperature + up._inv_λ) .* lnπ_θ
        return gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st, state_t
end

function _fkl_ll_grad(
    up::ContinuousRKLKL,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,                            # policy model
    π_θ,                            # policy model parameters
    π_st,                           # policy model state
    qf::Q,
    qf_f,                           # q function model
    qf_θ,                           # q function model parameters
    qf_st,                          # q function model state
    state_batch::AbstractArray,     # Must be >= 2D
    θ_t,                            # Previous policy parameters,
    state_t,                           # Previous policy state
)
    error("not implemented")
    batch_size = size(state_batch)[end]
    state_size = size(state_batch)[begin:end-1]
    num_samples = up._num_samples

    # Estimate the approximate state value for each state using v(s) = 𝔼[q(s, A)]
    if up._baseline_actions > 0
        v, π_st, qf_st = _get_baseline(
            rng, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, state_batch,
            up._baseline_actions,
        )
    end

    ∇π = gradient(π_θ) do π_θ
        # Sample actions for each state in the batch, needed to estimate the gradient in
        # each state: ∇J = 𝔼[q(S, A) - τ*ln(π(A∣S)]
        n_entropy_samples = num_samples
        actions, lnπ_t, state_t = sample_with_logprob(
            π, rng, π_f, θ_t, state_t, state_batch;
            num_samples=(num_samples + n_entropy_samples),
        )
        entropy_π_t = -mean(lnπ_t[end-n_entropy_samples+1:end, :]; dims=1)
        lnπ_t = lnπ_t[begin:begin+num_samples-1, :]

        actions = actions[:, begin:begin+num_samples-1, :]

        action_size = size(actions, 1)
        actions = reshape(actions, :, batch_size)

        # Repeat states for each action sample in each state
        states = repeat(
            state_batch;
            inner = (ones(Int, length(state_size))..., num_samples),
        )

        # Calculate the advantage
        q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states, actions; reduct=true)
        q = reshape(q, num_samples, batch_size)
        adv = up._baseline_actions > 0 ? q .- v : q

        lnπ_t = reshape(lnπ_t, size(adv))
        lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st, states, actions)
        lnπ_θ = reshape(lnπ_θ, size(adv))

        ς = ChainRulesCore.ignore_derivatives(
            adv .- up._temperature .* (lnπ_t .+ entropy_π_t)
        )

        -gpu_mean((ς .+ up._inv_λ) .* lnπ_θ)
    end

    return ∇π, π_st, qf_st, state_t
end
