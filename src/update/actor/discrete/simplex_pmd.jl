struct SimplexPMD <: AbstractActorUpdate
    _λ::Float32

    function SimplexPMD(λ)
        return new(λ)
    end
end

function setup(
    up::SimplexPMD,
    env::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
)::UpdateState{SimplexPMD}
    assert_uniform(π, π_θ)
    return UpdateState(up, nothing, NamedTuple())
end

# Constructor to satisfy common `setup` API for simplex algorithms
function setup(
    up::SimplexPMD,
    env::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
    ::Nothing,      # policy optimizer
    ::AbstractRNG;
)::UpdateState{SimplexPMD}
    return setup(up, env, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st)
end

function update(
    st::UpdateState{SimplexPMD},
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
)
    s_t = only(s_t)
    up = st._update

    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, s_t)

    π_k, π_st = prob(π, π_f, π_θ, π_st, s_t)
    π_kp1 = exp.(up._λ .* q) .* π_k
    π_kp1 ./= sum(π_kp1)

    π_θ = setcol(π_f, s_t, π_kp1, π_θ)

    return UpdateState(
        st._update,
        st._optim,
        st._state,
    ), π_θ, π_st, qf_st
end

function update(
    st::UpdateState{SimplexPMD},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    states,             # THIS SHOULD BE REMOVED
)
    if states !== nothing
        @info "ignoring states..." maxlog=1
    end

    up = st._update

    q, qf_st = predict(qf, qf_f, qf_θ, qf_st)

    π_k = π_θ.layer_1
    π_kp1 = exp.(up._λ .* q) .* π_k
    π_kp1 ./= sum(π_kp1; dims=1)

    # @show π_kp1
    π_θ = set(π_f, π_θ, π_kp1)

    return UpdateState(
        st._update,
        st._optim,
        st._state,
    ), π_θ, π_st, qf_st
end
