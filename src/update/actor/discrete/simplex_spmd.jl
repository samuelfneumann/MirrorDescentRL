struct SimplexSPMD <: AbstractActorUpdate
    _adaptive_step_size::Bool
    _max_step_size::Float32
end

function SimplexSPMD(;adaptive_step_size::Bool, max_step_size::Number = -1f0)
    return SimplexSPMD(adaptive_step_size, max_step_size)
end

function setup(
    up::SimplexSPMD,
    env::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
)::UpdateState{SimplexSPMD}
    assert_uniform(π, π_θ)
    return UpdateState(up, nothing, NamedTuple())
end

# Constructor to satisfy common `setup` API for simplex algorithms
function setup(
    up::SimplexSPMD,
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
)::UpdateState{SimplexSPMD}
    return setup(up, env, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st)
end

function update(
    st::UpdateState{SimplexSPMD},
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

    π_k, π_st = prob(π, π_f, π_θ, π_st, s_t)
    q_k, qf_st = predict(qf, qf_f, qf_θ, qf_st, s_t)
    v_k = π_k' * q_k

    A_k = q_k .- v_k
    A_k = A_k[begin+1:end, :]
    π_k = π_k[begin+1:end, :]

    neg_ind = A_k .< 0
    neg_lim = abs.(π_k[neg_ind] ./ A_k[neg_ind])

    pos_ind = A_k .> 0
    pos_lim = abs.((1 .- π_k[pos_ind]) ./ (
        A_k[pos_ind] .* π_k[pos_ind]
    ))

    μ = minimum([1f0, neg_lim..., pos_lim...])
    if up._max_step_size > 0f0
        μ = clamp.(μ, 0f0, up._max_step_size)
    end

    π_kp1 = π_k .* (1 .+ μ .* A_k)
    π_kp1 = vcat(1 .- sum(π_kp1), π_kp1)

    π_θ = setcol(π_f, s_t, π_kp1, π_θ)

    return UpdateState(
        st._update,
        st._optim,
        st._state,
    ), π_θ, π_st, qf_st
end

function update(
    st::UpdateState{SimplexSPMD},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    states,             # Must be >= 2D
)
    if states !== nothing
        @info "ignoring states..." maxlog=1
    end

    up = st._update

    π_k = π_θ.layer_1
    q_k, qf_st = predict(qf, qf_f, qf_θ, qf_st)
    v_k = sum(π_k .* q_k; dims=1)

    A_k = (q_k .- v_k)
    A_k = A_k[begin+1:end, :]
    π_k = π_k[begin+1:end, :]

    flat_A_k = [A_k...]
    flat_π_k = [π_k...]

    neg_ind = flat_A_k .< 0
    neg_lim = abs.([flat_π_k...][neg_ind] ./ [flat_A_k...][neg_ind])

    pos_ind = flat_A_k .> 0
    pos_lim = abs.((1 .- [flat_π_k...][pos_ind]) ./ (
        [flat_A_k...][pos_ind] .* [flat_π_k...][pos_ind]
    ))

    μ = if !up._adaptive_step_size
        minimum([1f0, neg_lim..., pos_lim...])
    else
        μ = ones(Float32, size(A_k))
        neg_ind = A_k .< 0
        pos_ind = A_k .> 0
        μ[neg_ind] .*= abs.(π_k[neg_ind] ./ A_k[neg_ind])
        μ[pos_ind] .*= (1 .- π_k[pos_ind]) ./ (A_k[pos_ind] .* π_k[pos_ind])
        μ = vcat(ones(Float32, (1, size(μ, 2))), μ)
        minimum(μ; dims=1)
    end

    if up._max_step_size > 0
        μ = clamp.(μ, 0f0, up._max_step_size)
    end

    π_kp1 = π_k .* (1 .+ μ .* A_k)
    π_kp1 = vcat(1 .- sum(π_kp1; dims=1), π_kp1)

    π_θ = set(π_f, π_θ, π_kp1)

    return UpdateState(
        st._update,
        st._optim,
        st._state,
    ), π_θ, π_st, qf_st
end
