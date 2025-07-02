struct SimplexRKL{S} <: AbstractActorUpdate
    _temperature::Float32
    _solver::S

    # Trick to ensure the policy is always slightly stochastic. This may be needed to
    # improve numerical stability when calculating entropy in the policy performance
    # gradient and when using soft action values, which also use entropy. If the policy
    # becomes deterministic, then the entropy will be -Inf.
    _ensure_stochastic::Bool
    _minp::Float32

    _use_baseline::Bool

    function SimplexRKL(τ, solver, ensure_stochastic, minp,  use_baseline)
        return new{typeof(solver)}(τ, solver, ensure_stochastic, minp, use_baseline)
    end
end

function SimplexRKL(
    τ; solver=SortSimplexProjection(), ensure_stochastic=true, minp=1f-7, use_baseline,
)
    return SimplexRKL(τ, solver, ensure_stochastic, minp, use_baseline)
end

function setup(
    up::SimplexRKL{S},
    ::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG;
) where {S}
    return UpdateState(
        up,
        optim,
        (rng = Lux.replicate(rng), optim=Optimisers.setup(optim, π_θ)),
    )
end

function update(
    st::UpdateState{SimplexRKL{S}},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    states,             # Must be >= 2D
) where {S}
    up = st._update
    rng = Lux.replicate(st._state.rng)
    optim_state = st._state.optim

    batch_size = size(states, 2)
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q

    # Construct gradient
    gs = zero(eltype(π_θ.layer_1), π_f)
    treemap!(gs) do g_i
        probs, π_st = up._temperature > 0f0 ? prob(π, π_f, π_θ, π_st) : (nothing, π_st)
        for i in 1:batch_size
            s_t = states[1, i]

            Δ = if up._temperature > 0f0
                adv[:, i] .- up._temperature .* log.(probs[:, s_t])
            else
                adv[:, i]
            end

            g_i[:, s_t] .-= (Δ ./ batch_size)
        end
        g_i
    end

    # Perform the mirror descent update, with an L2 norm mirror map
    optim_state, π_θ = Optimisers.update(optim_state, π_θ, gs)

    # Project back onto the simplex
    π_θ = treemap!(π_θ) do θ
        θ = simplex_project!(θ, up._solver)
        _ensure_stochastic!(π, π_f, θ; minp=up._minp)
    end

    return UpdateState(
        st._update,
        st._optim,
        ( rng = rng, optim=optim_state)
    ), π_θ, π_st, qf_st
end
