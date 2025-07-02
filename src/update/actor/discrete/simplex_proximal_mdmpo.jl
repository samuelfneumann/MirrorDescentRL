# Uses a minibatch style update
struct SimplexProximalMDMPO{S} <: AbstractActorUpdate
    _solver::S
    _kl_policy_coeff::Float32

    _ensure_stochastic::Bool
    _minp::Float32

    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature

    _inv_λ::Float32     # Inverse stepsize for mirror descent (functional) update
    _num_md_updates::Int

    _use_baseline::Bool

    # For continuous actions, n is the number of samples to use
    # For discrete actions, n is the number of actions
    function SimplexProximalMDMPO(
        kl_policy_coeff::Real, τ::Real, num_md_updates::Int, md_λ::Real, solver,
        ensure_stochastic, minp::AbstractFloat, use_baseline::Bool,
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"

        S = typeof(solver)
        return new{S}(
            solver, kl_policy_coeff, ensure_stochastic, minp, τ, inv(md_λ), num_md_updates,
            use_baseline,
        )
    end
end

function SimplexProximalMDMPO(
    kl_policy_coeff::Real, τ::Real, num_md_updates::Int, λ::Real;
    solver=SortSimplexProjection(), ensure_stochastic=true, minp=1f-7, use_baseline,
)
    return SimplexProximalMDMPO(
        kl_policy_coeff, τ, num_md_updates, λ, solver, ensure_stochastic, minp,
        use_baseline,
    )
end

function setup(
    up::SimplexProximalMDMPO,
    ::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG,
)
    return UpdateState(
        up,
        optim,
        (
            π_optim = Optimisers.setup(optim, π_θ),
            # Previous policy parameters for the KL update
            θ_t = π_θ,    # These are immutable
            state_t = π_st,  # These are immutable
            current_update = 1,
        )
    )
end

function update(
    st::UpdateState{SimplexProximalMDMPO{S}},
    π::SimplexPolicy,
    π_f::Tabular,   # actor policy model
    π_θ,            # actor policy model parameters
    π_st,           # actor policy model state
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
) where {S}
    up = st._update

    # Frozen current policy parameters, must stay fixed during the MD update and only update
    # every up._num_md_updates
    θ_t = st._state.θ_t
    # State of the current policy, which will change during the MD update
    state_t = st._state.state_t

    # Compute advantages
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q
    @assert ndims(adv) == 2
    batch_size = size(adv, 2)

    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, states)
    π_t = exp.(lnπ_t)

    # Compute KL policy in πₜ
    π_KL_logits = adv ./ up._kl_policy_coeff
    π_KL_logits_exp = exp.(π_KL_logits .- maximum(π_KL_logits; dims=1))
    π_KL_numerator = π_KL_logits_exp .* π_t
    π_KL = π_KL_numerator ./ sum(π_KL_numerator; dims=1)

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st)
    gs = zero(Float32, π_f)
    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]

            lr_term = π_KL[:, i] ./ π_t[:, i]
            entropy_t_term = (up._τ - up._inv_λ) .* lnπ_t[:, i]
            entropy_term = up._inv_λ .* lnπ_θ[:, s_t]

            scale = lr_term .- entropy_t_term .- entropy_term

            ∇π_θ = _∇_simplex_tabular(π_θ.layer_1, s_t; sum_over_actions=false)
            ∇π_θ .*= reshape(scale, 1, :)
            ∇π_θ_term = dropdims(sum(∇π_θ; dims=2); dims=2)

            g_i[:, s_t] .-= (∇π_θ_term ./ batch_size)
            end
        g_i
    end

    optim_state = st._state.π_optim
    optim_state, π_θ = Optimisers.update(optim_state, π_θ, gs)

    # Project back to the simplex
    π_θ = treemap(π_θ) do θ
        project_ensure_stochastic(
            st, π, π_f, θ, states; ensure_stochastic=up._ensure_stochastic, minp=up._minp,
        )
    end

    next_update = mod(st._state.current_update, up._num_md_updates) + 1
    return UpdateState(
        st._update,
        st._optim,
        (
            π_optim = optim_state,
            θ_t = next_update == 1 ? π_θ : θ_t,
            state_t = next_update == 1 ? π_st : state_t,
            current_update = next_update,
        ),
    ), π_θ, π_st, qf_st
end
