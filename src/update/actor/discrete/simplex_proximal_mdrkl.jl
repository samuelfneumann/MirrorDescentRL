# Uses a minibatch style update
struct SimplexProximalMDRKL{S} <: AbstractActorUpdate
    _solver::S

    _ensure_stochastic::Bool
    _minp::Float32

    # Actor Policy Entropy Regularization
    _temperature::Float32         # Actor Temperature

    _inv_λ::Float32     # Inverse stepsize for mirror descent (functional) update
    _num_md_updates::Int

    _use_baseline::Bool

    function SimplexProximalMDRKL(
        τ::Real, num_md_updates::Int, md_λ::Real, solver, ensure_stochastic,
        minp::Real, use_baseline::Bool,
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"

        S = typeof(solver)
        return new{S}(
            solver, ensure_stochastic, minp, τ, inv(md_λ), num_md_updates, use_baseline,
        )
    end
end

function SimplexProximalMDRKL(
    τ::Real, num_md_updates::Int, λ::Real;
    use_baseline::Bool, solver=SortSimplexProjection(), ensure_stochastic=true, minp=1f-7,
)
    return SimplexProximalMDRKL(
        τ, num_md_updates, λ, solver, ensure_stochastic, minp, use_baseline,
    )
end

function setup(
    up::SimplexProximalMDRKL,
    env::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    _::AbstractRNG,
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
    st::UpdateState{SimplexProximalMDRKL{S}},
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

    batch_size = size(states)[end]
    n_actions = size(π_f, 1)

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = zero(Float32, π_f)

    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]
            q_t, qf_st = predict(qf, qf_f, qf_θ, qf_st, states[:, i])
            if up._use_baseline
                v_t = mean(q_t; dims=1)
                q_t .-= v_t
            end

            lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st, [s_t])
            lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, [s_t])

            scale = (
                q_t .-
                up._temperature * (1 + up._inv_λ) .* lnπ_t .+
                (up._temperature * up._inv_λ) .* lnπ_θ
            )

            ∇loss_θ = scale[1] * _∇_simplex_tabular(π_θ.layer_1, s_t, 1)
            for a_t in 2:n_actions
                ∇loss_θ .+= scale[a_t] .* _∇_simplex_tabular(π_θ.layer_1, s_t, a_t)
            end

            g_i[:, s_t] .-= (∇loss_θ ./ batch_size)
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
