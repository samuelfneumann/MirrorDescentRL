# Uses a minibatch style update
struct SimplexProximalMDCCEM{S} <: AbstractActorUpdate
    _solver::S

    _ensure_stochastic::Bool
    _minp::Float32

    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature

    _inv_λ::Float32     # Inverse stepsize for mirror descent (functional) update
    _num_md_updates::Int

    function SimplexProximalMDCCEM(
        τ::Real, num_md_updates::Int, md_λ::Real, solver, ensure_stochastic,
        minp::Real,
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"

        S = typeof(solver)
        return new{S}(solver, ensure_stochastic, minp, τ, inv(md_λ), num_md_updates)
    end
end

function SimplexProximalMDCCEM(
    τ::Real, num_md_updates::Int, λ::Real;
    solver=SortSimplexProjection(), ensure_stochastic=true, minp=1f-7,
)
    return SimplexProximalMDCCEM(τ, num_md_updates, λ, solver, ensure_stochastic, minp)
end

function setup(
    up::SimplexProximalMDCCEM,
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
    st::UpdateState{SimplexProximalMDCCEM{S}},
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

    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    batch_size = size(states)[end]

    # Find the indices of the actions of maximal value
    ind = mapslices(x -> sortperm(x; rev=true), q; dims=1)
    top_actions = [ind[1, j] for j in 1:batch_size]

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = zero(Float32, π_f)
    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t)
    π_t, state_t = prob(π, π_f, θ_t, state_t)
    lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st)
    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]
            a_t = top_actions[i]

            lr_term = _∇_simplex_tabular(π_θ.layer_1, s_t, a_t) ./ π_t[a_t, s_t]

            ∇π_θ = _∇_simplex_tabular(π_θ.layer_1, s_t; sum_over_actions=false)
            kl_entropy_term = (
                (up._τ - up._inv_λ) .* lnπ_t[:, s_t] +
                up._inv_λ .* lnπ_θ[:, s_t]
            )
            kl_entropy_term = reshape(kl_entropy_term, 1, :)
            ∇π_θ .*= kl_entropy_term
            ∇π_θ_term = dropdims(sum(∇π_θ; dims=2); dims=2)

            g_i[:, s_t] .-= ((lr_term .- ∇π_θ_term) ./ batch_size)
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


function project_ensure_stochastic(
    st::UpdateState{U,<:Descent}, π::AbstractPolicy, π_f::Tabular, π_θ,
    states_used_in_update; ensure_stochastic, minp,
) where {U}
    return project_ensure_stochastic_over(
        st, π, π_f, π_θ, states_used_in_update; ensure_stochastic, minp,
    )
end

function project_ensure_stochastic(
    st, π::AbstractPolicy, π_f::Tabular, π_θ, states_used_in_update; ensure_stochastic,
    minp,
)
    return project_ensure_stochastic_over(
        st, π, π_f, π_θ, 1:size(π_f, 2); ensure_stochastic, minp,
    )
end

function project_ensure_stochastic_over(
    st, π::AbstractPolicy, π_f::Tabular, π_θ, states; ensure_stochastic, minp,
)
    up = st._update
    for state in states
        π_θ[:, state] .= simplex_project(π_θ[:, state], up._solver)

        if ensure_stochastic
            _ensure_stochastic!(π, π_f, π_θ, state; minp)
        end
    end
    π_θ
end
