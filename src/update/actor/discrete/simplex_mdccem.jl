"""
    SimplexMDCCEM <: AbstractActorUpdate

SimplexMDCCEM implements the SimplexMDCCEM algorithm using the negative entropy mirror map. This sturct
uses MiniBatch-style updates
"""
struct SimplexMDCCEM <: AbstractActorUpdate
    _τ::Float32

    # Trick to ensure the policy is always slightly stochastic. This may be needed to
    # improve numerical stability when calculating entropy in the policy performance
    # gradient and when using soft action values, which also use entropy. If the policy
    # becomes deterministic, then the entropy will be -Inf.
    _ensure_stochastic::Bool
    _minp::Float32
end

function SimplexMDCCEM(τ; ensure_stochastic=true, minp=1f-7)
    return SimplexMDCCEM(τ, ensure_stochastic, minp)
end

function setup(
    up::SimplexMDCCEM,
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
)::UpdateState{SimplexMDCCEM}
    assert_uniform(π, π_θ)
    return UpdateState(
        up,
        optim,
        (rng = Lux.replicate(rng), optim = Optimisers.setup(optim, π_θ),),
    )
end

function update(
    st::UpdateState{SimplexMDCCEM},
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
    up = st._update
    rng = st._state.rng
    optim_state = st._state.optim

    batch_size = size(states, 2)
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)

    # Find action of maximal value, breaking ties randomly
    a_maxs = mapslices(x -> argmax_break_ties_randomly(rng, x), q; dims=1)

    # Construct gradient
    gs = zero(eltype(π_θ.layer_1), π_f)
    treemap!(gs) do g_i
        probs, π_st = prob(π, π_f, π_θ, π_st)
        for i in 1:batch_size
            s_t = states[1, i]
            a_t = a_maxs[i]
            a_t_prob = probs[a_t, s_t]

            if up._τ > 0f0
                Δ =  up._τ .* log.(probs[:, s_t])
                Δ[a_t] -= inv(a_t_prob)
                g_i[:, s_t] .+= (Δ ./ batch_size)
            else
                g_i[a_t, s_t] -= (inv(a_t_prob) ./ batch_size)
            end
        end
        g_i
    end

    ####################################################################
    # Perform the mirror ascent update
    ####################################################################
    # Set input
    x_i = π_θ
    # Transform to dual space
    x̂_i = treemap(x -> log.(x) .+ 1, x_i)
    # Step in dual space: ŷ_ip1 = x̂_i .+ λ .* g_i
    optim_state, ŷ_ip1 = Optimisers.update(optim_state, x̂_i, gs)
    # Project back to mirror map domain
    y_ip1 = treemap(x -> exp.(x .- 1), ŷ_ip1)
    # Project back to input (simplex) space
    x_ip1 = treemap(y_ip1) do y
        infs = isinf.(y)
        correct_infs = any(infs)
        inf_rows = dropdims(any(infs; dims=1); dims=1)

        if correct_infs
            # For rows with infs, project infs to 1 and all other values to 0. The
            # normalization following will ensure that the simplex vectors are normalized if
            # there are multiple infs.
            y[:, inf_rows] .= zero(eltype(y))
            y[infs] .= one(eltype(y))
        end

        # Project back to the simplex
        normalizer = sum(abs.(y); dims=1)
        y ./= normalizer

        _ensure_stochastic!(π, π_f, y; minp=up._minp)
    end

    π_θ = x_ip1

    return UpdateState(
        st._update,
        st._optim,
        (rng = rng, optim = optim_state,),
    ), π_θ, π_st, qf_st
end
