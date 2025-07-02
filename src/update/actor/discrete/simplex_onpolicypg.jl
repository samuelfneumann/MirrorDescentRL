struct OnPolicySimplexPG{S} <: AbstractActorUpdate
    _solver::S
    _γ::Float32

    function OnPolicySimplexPG(solver, γ)
        return new{typeof(solver)}(solver, γ)
    end
end

function OnPolicySimplexPG(;solver=SortSimplexProjection(), γ)
    return OnPolicySimplexPG(solver, γ)
end


# Constructor to satisfy common `setup` API for simplex algorithms
function setup(
    up::OnPolicySimplexPG,
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
        up, optim, (optim=Optimisers.setup(optim, π_θ),),
    )
end

function update(
    st::UpdateState{OnPolicySimplexPG{S}},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    s_t::Matrix{<:Integer},             # Must be >= 2D
    term::Bool,
) where {S}
    s_t = only(s_t)
    up = st._update

    # Construct the gradient
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, s_t)

    # Construct policy gradient
    gs = spzeros(eltype(π_θ.layer_1), π_f)
    treemap!(gs) do g_i
        g_i[:, s_t] .= -q
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
        (optim=optim_state,),
    ), π_θ, π_st, qf_st
end


function update(
    st::UpdateState{OnPolicySimplexPG{S}},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    states,
) where {S}
    @assert size(states[begin]) == (1,)

    up = st._update

    # Construct the gradient
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st)

    # Construct on-policy distribution
    # TODO: instead of using a Monte-Carlo estimate, we could instead use
    # μ^⊤ inv(1 - P(π)). This is the probability of starting in a state s₀ ∼ μ and
    # transitioning to each state from s₀
    d = convert.(Float32, map(i -> count(==(i), states), 1:size(q, 2)))
    d ./= length(states)
    d = reshape(d, (1, size(d)...))

    # Construct policy gradient
    gs = zero(eltype(π_θ.layer_1), π_f)
    treemap!(gs) do g_i
        g_i .= -inv(1 - up._γ) .* (q .* d)
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
        (optim=optim_state,),
    ), π_θ, π_st, qf_st
end

