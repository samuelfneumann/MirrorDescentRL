const _simplex_proj_atol = 1f-7

struct SimplexMPO{S} <: AbstractActorUpdate
    _solver::S

    _kl_policy_coeff::Float32
    _temperature::Float32

    # Trick to ensure the policy is always slightly stochastic. This may be needed to
    # improve numerical stability when calculating entropy in the policy performance
    # gradient and when using soft action values, which also use entropy. If the policy
    # becomes deterministic, then the entropy will be -Inf.
    _ensure_stochastic::Bool
    _minp::Float32

    _use_baseline::Bool

    function SimplexMPO(
        kl_policy_coeff::Real, τ::Real, solver, ensure_stochastic, minp, use_baseline,
    )
        return new{typeof(solver)}(
            solver, kl_policy_coeff, τ, ensure_stochastic, minp, use_baseline,
        )
    end
end

function SimplexMPO(
    kl_policy_coeff, τ; solver=SortSimplexProjection(), ensure_stochastic=true, minp=1f-7,
    use_baseline,
)
    return SimplexMPO(kl_policy_coeff, τ, solver, ensure_stochastic, minp, use_baseline)
end

function setup(
    up::SimplexMPO,
    ::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG;
)
    return UpdateState(
        up,
        optim,
        (optim = Optimisers.setup(optim, π_θ),),
    )
end

function update(
    st::UpdateState{SimplexMPO{S}},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    state::Int,
) where {S}
    up = st._update
    optim_state = st._state.optim

    π_θ, π_st, qf_st, optim_state, = _update(
        up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, state, optim_state,
    )

    return UpdateState(
        st._update,
        st._optim,
        (optim = optim_state,),
    ), π_θ, π_st, qf_st
end

function update(
    st::UpdateState{SimplexMPO{S}},
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
    optim_state = st._state.optim

    # Compute advantages
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q
    @assert ndims(adv) == 2

    # Compute KL policy in πₜ
    probs, π_st = prob(π, π_f, π_θ, π_st)
    π_KL_logits = adv ./ up._kl_policy_coeff
    π_KL_logits_exp = exp.(π_KL_logits .- maximum(π_KL_logits; dims=1))
    π_KL_numerator = π_KL_logits_exp .* probs
    π_KL = π_KL_numerator ./ sum(π_KL_numerator; dims=1)

    # Construct gradient
    batch_size = size(states, 2)
    gs = zero(Float32, π_f)
    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]

            grad = -π_KL[:, s_t] .* inv.(probs[:, s_t])
            if up._temperature > 0
                grad .+= up._temperature .* log.(probs[:, s_t])
            end

            g_i[:, s_t] .+= (grad ./ batch_size)
        end
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
        (optim = optim_state,),
    ), π_θ, π_st, qf_st
end
