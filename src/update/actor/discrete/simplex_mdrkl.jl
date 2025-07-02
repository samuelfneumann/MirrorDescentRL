"""
    SimplexMDRKL <: AbstractActorUpdate

SimplexMDRKL implements the closed-from SimplexMDRKL algorithm using the negative entropy mirror map. This
sturct uses a Minibatch-style update.
"""
struct SimplexMDRKL <: AbstractActorUpdate
    _temperature::Float32

    # Trick to ensure the policy is always slightly stochastic. This may be needed to
    # improve numerical stability when calculating entropy in the policy performance
    # gradient and when using soft action values, which also use entropy. If the policy
    # becomes deterministic, then the entropy will be -Inf.
    _ensure_stochastic::Bool
    _minp::Float32

    _use_baseline::Bool

    function SimplexMDRKL(τ, ensure_stochastic, minp, use_baseline)
        return new(τ, ensure_stochastic, minp, use_baseline)
    end
end

function SimplexMDRKL(τ; ensure_stochastic=true, minp=1f-7, use_baseline::Bool)
    return SimplexMDRKL(τ, ensure_stochastic, minp, use_baseline)
end

function setup(
    up::SimplexMDRKL,
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
    rng::AbstractRNG;
)::UpdateState{SimplexMDRKL}
    assert_uniform(π, π_θ)
    return UpdateState(
        up,
        optim,
        (rng = Lux.replicate(rng), optim=Optimisers.setup(optim, π_θ)),
    )
end

function update(
    st::UpdateState{SimplexMDRKL},
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
        ( rng = rng, optim=optim_state)
    ), π_θ, π_st, qf_st
end
