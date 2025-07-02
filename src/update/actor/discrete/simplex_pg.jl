struct SimplexPG{S} <: AbstractActorUpdate
    _solver::S
    _γ::Float32

    function SimplexPG(solver, γ::AbstractFloat)
        return new{typeof(solver)}(solver, γ)
    end
end

function SimplexPG(;solver=SortSimplexProjection(), γ)
    return SimplexPG(solver, γ)
end


# Constructor to satisfy common `setup` API for simplex algorithms
function setup(
    up::SimplexPG,
    env::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG;
)
    assert_uniform(π, π_θ)
    return UpdateState(
        up, optim, (optim=Optimisers.setup(optim, π_θ), γ_scale=1f0),
    )
end

function update(
    st::UpdateState{SimplexPG{S}},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    s_t::Matrix{<:Integer},
    term::Bool
) where {S}
    s_t = only(s_t)

    up = st._update
    γ_scale = st._state.γ_scale

    # Construct the gradient
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, s_t)

    # Construct policy gradient
    gs = spzeros(eltype(π_θ.layer_1), π_f)
    treemap!(gs) do g_i
        #? Question: since we update the policy while following the policy (to update the
        # critic) for N steps, does this mean that the scale should have a much larger
        # power?
        g_i[:, s_t] .= -γ_scale .* q
    end

    optim = st._optim
    optim_state = st._state.optim
    optim_state, π_θ = Optimisers.update(optim_state, π_θ, gs)

    # Simplex projection
    π_θ = treemap!(π_θ) do θ
        simplex_project!(θ, up._solver)
    end

    return UpdateState(
        st._update,
        st._optim,
        (optim=optim_state, γ_scale=term ? 1f0 : γ_scale * up._γ),
    ), π_θ, π_st, qf_st
end


function update(
    st::UpdateState{SimplexPG{S}},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    d_μ,
) where {S}
    up = st._update

    # Construct the gradient
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st)

    # Construct policy gradient
    gs = zero(eltype(π_θ.layer_1), π_f)
    treemap!(gs) do g_i
        g_i .= - inv(1 - up._γ) .* (q .* reshape(d_μ, (1, length(d_μ))))
    end

    optim = st._optim
    optim_state = st._state.optim
    optim_state, π_θ = Optimisers.update(optim_state, π_θ, gs)

    # Simplex projection
    π_θ = treemap!(π_θ) do θ
        simplex_project!(θ, up._solver)
    end

    return UpdateState(
        st._update,
        st._optim,
        (optim=optim_state, γ_scale = st._state.γ_scale),
    ), π_θ, π_st, qf_st
end

