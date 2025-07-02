struct Sarsa{S<:Tuple} <: AbstractActionValueCriticUpdate
    # S should be an empty tuple or a tuple of AbstractBellmanRegularisers
    _reg::S

    function Sarsa(reg::S) where {S}
        new{S}(reg)
    end
end

Sarsa() = Sarsa(tuple())
Sarsa(reg::AbstractBellmanRegulariser) = Sarsa((reg,))

function setup(
    up::Sarsa{S},
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState where {S}
    # Initialize the Bellman regularizers
    reg_st = [
        setup(reg, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, Lux.replicate(rng))
        for reg in up._reg
    ]

    return UpdateState(
        up,
        optim,
        (
            optim = Optimisers.setup(optim, qf_θ),
            rng = Lux.replicate(rng),
            reg_st = tuple(reg_st...),
        ),
    )
end

function setup(
    up::Sarsa{S},
    π::SimplexPolicy,
    π_f::Tabular,
    π_θ,
    π_st,
    qf::DiscreteQ,
    qf_f::Tabular,
    qf_θ,
    qf_st,
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState where {S}
    # Initialize the Bellman regularizers
    # @assert length(up._reg) == 0 "Bellman regularizers not yet implemented for tabular"

    reg_st = [
        setup(reg, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, Lux.replicate(rng))
        for reg in up._reg
    ]

    return UpdateState(
        up,
        optim,
        (
            optim = Optimisers.setup(optim, qf_θ),
            rng = Lux.replicate(rng),
            reg_st = tuple(reg_st...),
        ),
    )
end

function update(
    st::UpdateState{<:Sarsa},
    π::AbstractDiscreteParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    qf_target_θ,
    qf_target_st,
    s_t,
    a_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    up = st._update
    rng = Lux.replicate(st._state.rng)

    # Compute the (regularised) target using the target network
    a_tp1, π_st = sample(π, rng, π_f, π_θ, π_st, s_tp1; num_samples=1)
    a_tp1 = a_tp1[1, :]
    q_tp1, qf_target_st = predict(qf, qf_f, qf_target_θ, qf_target_st, s_tp1, a_tp1)

    target = if length(up._reg) == 0
        reg_states = tuple()
        r_tp1 .+ γ_tp1 .* q_tp1
    else
        ####################################################################
        # Accumulate the regularization to add to the bellman operator
        ####################################################################
        accum_reg_term, π_st, qf_st, reg_st = regularise(
            st._state.reg_st[1], s_tp1, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st,
        )
        reg_states = RegulariserState[reg_st]

        for i in 2:length(st._state.reg_st)
            reg_term, π_st, qf_st, reg_st = regularise(
                st._state.reg_st[i], s_tp1, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st,
            )
            accum_reg_term .+= reg_term
            push!(reg_states, reg_st)
        end
        ####################################################################
        reg_states = tuple(reg_states...)

        r_tp1 .+ γ_tp1 .* (q_tp1 .+ accum_reg_term)
   end

    ∇q_θ = gradient(qf_θ) do θ
        q_t, qf_st = predict(qf, qf_f, θ, qf_st, s_t, a_t)
        gpu_mean((q_t .- target) .^ 2)
    end

    q_optim_state = st._state.optim
    q_optim_state, qf_θ = Optimisers.update(q_optim_state, qf_θ, ∇q_θ[1])

    return UpdateState(
        st._update,
        st._optim,
        (
            optim = q_optim_state,
            rng = rng,
            reg_st = reg_states,
        ),
    ), qf_θ, qf_st, qf_target_st, π_st
end

function update(
    st::UpdateState{<:Sarsa},
    π::AbstractContinuousParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    qf::Q,
    qf_f,
    qf_θ,
    qf_st,
    qf_target_θ,
    qf_target_st,
    s_t,
    a_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    up = st._update
    rng = Lux.replicate(st._state.rng)

    # Compute the (regularised) target using the target network
    as_tp1, lnπs_tp1, π_st = sample_with_logprob(
        π, rng, π_f, π_θ, π_st, s_tp1; num_samples=1,
    )
    a_tp1 = as_tp1[:, 1, :] # (action_dims, num_samples=1, batch_size)
    lnπ_tp1 = lnπs_tp1[1, :] # (num_samples=1, batch_size)
    q_tp1, qf_target_st = predict(qf, qf_f, qf_target_θ, qf_target_st, s_tp1, a_tp1)

    target = if length(up._reg) == 0
        reg_states = tuple()
        r_tp1 .+ γ_tp1 .* q_tp1
    else
        ####################################################################
        # Accumulate the regularization to add to the Bellman operator
        ####################################################################
        accum_reg_term, π_st, qf_st, reg_st = regularise(
            st._state.reg_st[1], s_tp1, π, π_f, π_θ, π_st, a_tp1, lnπ_tp1, qf, qf_f, qf_θ,
            qf_st,
        )
        reg_states = RegulariserState[reg_st]

        for i in 2:length(st._state.reg_st)
            reg_term, π_st, qf_st, reg_st = regularise(
                st._state.reg_st[i], s_tp1, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st,
            )
            accum_reg_term .+= reg_term
            push!(reg_states, reg_st)
        end
        ####################################################################
        reg_states = tuple(reg_states...)

        r_tp1 .+ γ_tp1 .* (q_tp1 .+ accum_reg_term)
    end

    ∇q = gradient(qf_θ) do θ
        q_t, qf_st = predict(qf, qf_f, θ, qf_st, s_t, a_t)
        gpu_mean((q_t .- target) .^ 2)
    end

    q_optim_state = st._state.optim
    q_optim_state, qf_θ = Optimisers.update(q_optim_state, qf_θ, only(∇q))

    return UpdateState(
        st._update,
        st._optim,
        (
            optim = q_optim_state,
            rng = rng,
            reg_st = reg_states,
        ),
    ), qf_θ, qf_st, qf_target_st, π_st
end

function update(
    st::UpdateState{<:Sarsa},
    π::SimplexPolicy,
    π_f::Tabular,
    π_θ,
    π_st,
    qf::DiscreteQ,
    qf_f::Tabular,
    qf_θ,
    qf_st,
    qf_target_θ,
    qf_target_st,
    s_t::Int,
    a_t::Int,
    r_tp1::Real,
    s_tp1::Int,
    γ_tp1::Real,
)
    update(
        st, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, qf_target_θ, qf_target_st, [s_t],
        [a_t], [r_tp1], [s_tp1], [γ_tp1],
    )
end

function update(
    st::UpdateState{<:Sarsa},
    π::Union{SimplexPolicy,SoftmaxPolicy},
    π_f::Tabular,
    π_θ,
    π_st,
    qf::DiscreteQ,
    qf_f::Tabular,
    qf_θ,
    qf_st,
    qf_target_θ,
    qf_target_st,
    s_t,
    a_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    up = st._update
    rng = Lux.replicate(st._state.rng)

    q_t, qf_st = qf(qf_f, qf_θ, qf_st, s_t, a_t)
    a_tp1, π_st = sample(π, π_f, π_θ, π_st, s_tp1; num_samples=1)
    q_tp1, qf_st = qf(qf_f, qf_θ, qf_st, s_tp1, a_tp1)

    target = if length(up._reg) == 0
        reg_states = tuple()
        r_tp1 + γ_tp1 .* q_tp1
    else
       ####################################################################
       # Accumulate the regularization to add to the bellman operator
       ####################################################################
        accum_reg_term, π_st, qf_st, reg_st = regularise(
            st._state.reg_st[1], s_tp1, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st,
        )
        reg_states = RegulariserState[reg_st]

        for i in 2:length(st._state.reg_st)
            reg_term, π_st, qf_st, reg_st = regularise(
                st._state.reg_st[i], s_tp1, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st,
            )
            accum_reg_term .+= reg_term
            push!(reg_states, reg_st)
        end
        ####################################################################
        reg_states = tuple(reg_states...)

        r_tp1 .+ γ_tp1 .* (q_tp1 .+ accum_reg_term)
    end

    δ = q_t - target

    # Construct gradient
    ∇q_θ = spzeros(qf_f)
    ∇q_θ = set(qf_f, ∇q_θ, a_t, s_t[1, :], δ)

    q_optim_state = st._state.optim
    q_optim_state, qf_θ = Optimisers.update(q_optim_state, qf_θ, ∇q_θ)

    return UpdateState(
        st._update,
        st._optim,
        (
            optim = q_optim_state,
            rng = rng,
            reg_st = reg_states,
        ),
    ), qf_θ, qf_st, qf_target_st, π_st
end
