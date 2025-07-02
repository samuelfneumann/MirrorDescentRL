"""
    struct BatchQAgent{P,PF,PS,Q,QF,QS,ER,RNG} <: AbstractAgent

A `BatchQAgent` is an actor-critic agent which uses an action-value critic and uses batch
updates.
"""
mutable struct BatchQAgent{Π,ΠM,ΠMΘ,ΠMS,ΠUS,Q,QM,QMΘ,QMS,QUS,ER,RNG,D} <: AbstractAgent where {
    Π<:AbstractParameterisedPolicy, # π
    ΠM,                             # π Model
    ΠMΘ,                            # π Model Parameters
    ΠMS,                            # π Model State
    ΠUS<:UpdateState,               # π Update State
    Q<:AbstractActionValueFunction,
    QM,                                                 # Value Function Model
    QMΘ,                                                # Value Function Model Parameters
    QMS,                                                # Value Function Model State
    QUS<:UpdateState{<:AbstractActionValueCriticUpdate},# Value Function Update State
    ER<:AbstractReplay,
    RNG<:AbstractRNG,
    D<:AbstractDevice,
}
    # Policy
    _π::Π
    _π_model::ΠM
    _π_θ::ΠMΘ
    _π_st::ΠMS
    _π_update_st::ΠUS
    _π_update_ratio::UpdateRatio

    # Q function
    _q̂::Q
    _q̂_model::QM
    _q̂_θ::QMΘ
    __q̂_θ::QMΘ
    _q̂_st::QMS
    _q̂_update_st::QUS
    _q̂_update_ratio::UpdateRatio

    _max_n_updates::UInt

    # Target Q function
    _q̂_target_θ::QMΘ
    _q̂_target_st::QMS
    _q̂_target_refresh_steps::Int
    _q̂_polyak_avg::Float32
    _use_target_nets::Bool

    _buffer::ER

    _batch_size::Int

    _current_step::UInt
    _current_critic_update::UInt
    _current_actor_update::UInt
    _steps_before_learning::Int

    _rng::RNG

    _is_training::Bool

    _device::D

    function BatchQAgent(
        seed::Integer,
        env::AbstractEnvironment,
        π::P,
        π_model::PM,
        π_optim,
        π_update,
        π_update_ratio,
        q̂::Q,
        q̂_model::QM,
        q̂_optim,
        q̂_update::AbstractActionValueCriticUpdate,
        q̂_update_ratio,
        q̂_target_refresh_steps,
        q̂_polyak_avg,
        buffer;
        batch_size,
        steps_before_learning,
        device,
    ) where {P,PM,Q,QM}
        rng = if device isa Lux.CUDADevice
            CUDA.RNG(seed)
        elseif device isa Lux.CPUDevice
            rng = Random.default_rng()
            Random.seed!(rng, seed)
        else
            error("unknown device $(typeof(device))")
        end

        # Move the policy to the correct device
        # π = π |> device
        _P = typeof(π)

        # Initialize policy and q function parameters and state
        π_θ, π_st = setup(Lux.replicate(rng), π_model)
        q̂_θ, q̂_st = setup(Lux.replicate(rng), q̂_model)

        # Initialize π update
        π_update_st = setup(
            π_update, env, π, π_model, π_θ, π_st, q̂, q̂_model, q̂_θ, q̂_st, π_optim,
            Lux.replicate(rng),
        )

        # Initialize q-function update
        q̂_update_st = setup(
            q̂_update, π, π_model, π_θ, π_st, q̂, q̂_model, q̂_θ, q̂_st, q̂_optim,
            Lux.replicate(rng),
        )

        # Ensure parameters for target nets are valid
        @assert q̂_target_refresh_steps > 0
        @assert 0 < q̂_polyak_avg <= 1

        # Initialize target nets, if required
        use_target_nets = q̂_target_refresh_steps != 1 || q̂_polyak_avg != 1f0
        q̂_target_θ, q̂_target_st = if use_target_nets
            deepcopy(q̂_θ), deepcopy(q̂_st)
        else
            q̂_θ, q̂_st
        end

        # buffer = buffer |> device

        π_update_ratio = UpdateRatio(π_update_ratio)
        q̂_update_ratio = UpdateRatio(q̂_update_ratio)
        max_n_updates = max(
            updates_per_step(π_update_ratio)[1], updates_per_step(q̂_update_ratio)[1],
        )

        PMΘ = typeof(π_θ)
        PMS = typeof(π_st)
        PUS = typeof(π_update_st)
        QMΘ = typeof(q̂_θ)
        QMS = typeof(q̂_st)
        QUS = typeof(q̂_update_st)
        ER = typeof(buffer)

        agent = new{
            _P,PM,PMΘ,PMS,PUS,Q,QM,QMΘ,QMS,QUS,ER,typeof(rng),typeof(device),
        }(
            π,
            π_model,
            π_θ,
            π_st,
            π_update_st,
            π_update_ratio,
            q̂,
            q̂_model,
            q̂_θ,
            deepcopy(q̂_θ),
            q̂_st,
            q̂_update_st,
            q̂_update_ratio,
            max_n_updates,
            q̂_target_θ,
            q̂_target_st,
            q̂_target_refresh_steps,
            q̂_polyak_avg,
            use_target_nets,
            buffer,
            batch_size,
            1,
            1,
            1,
            steps_before_learning,
            rng,
            true,
            device,
        )

        # Set the agent to training mode
        train!(agent)
        return agent
    end
end

function BatchQAgent(
    seed::Integer,
    env::AbstractEnvironment,
    π::P,
    π_model::PM,
    π_optim,
    π_update,
    π_update_ratio,
    q̂::Q,
    q̂_model::QM,
    q̂_optim,
    q̂_update,
    q̂_update_ratio,
    buffer::ER;
    batch_size,
    steps_before_learning,
) where {P,PM,Q,QM,ER}
    return BatchQAgent(
        seed, env, π, π_model, π_optim, π_update, π_update_ratio, q̂, q̂_model,
        q̂_optim, q̂_update, q̂_update_ratio, 1, 1f0, buffer, batch_size,
        steps_before_learning,
    )
end

device(ag::BatchQAgent) = ag._device

function train!(agent::BatchQAgent)::Nothing
    agent._is_training = true
    agent._π_st = train(agent._π_model, agent._π_st)
    agent._q̂_st = train(agent._q̂_model, agent._q̂_st)
    return nothing
end

function eval!(agent::BatchQAgent)::Nothing
    agent._is_training = false
    agent._π_st = eval(agent._π_model, agent._π_st)
    agent._q̂_st = eval(agent._q̂_model, agent._q̂_st)
    return nothing
end

function select_action(ag::BatchQAgent{<:AbstractContinuousParameterisedPolicy}, s_t)
    s_t = s_t |> device(ag)
    if !ag._is_training
        error("not implemented")
    else
        action, ag._π_st = sample(
            ag._π, ag._rng, ag._π_model, ag._π_θ, ag._π_st,
            s_t,
        )
        return action[:, 1]
    end
end

function select_action(
    ag::BatchQAgent{<:AbstractDiscreteParameterisedPolicy}, s_t::AbstractVector,
)
    s_t = s_t |> device(ag)
    if !ag._is_training
        error("not implemented")
    else
        action, ag._π_st = sample(
            ag._π, ag._rng, ag._π_model, ag._π_θ, ag._π_st,
            s_t,
        )
        return action
    end
end

function start!(agent::BatchQAgent, s_0)::Nothing
    if !agent._is_training
        @warn "calling start! on an agent in evaluation mode is a no-op, returning..."
        return
    end
end

function step!(ag::BatchQAgent, s_t, a_t, r_tp1, s_tp1, γ_tp1)::Nothing
    if !ag._is_training
        @warn "calling step! on an ag in evaluation mode is a no-op, returning..."
        return
    end

    s_t = s_t |> device(ag)
    a_t = a_t |> device(ag)
    s_tp1 = s_tp1 |> device(ag)

    # Add transition to the replay buffer
    push!(ag._buffer, s_t, a_t, [r_tp1] |> device(ag), s_tp1, [γ_tp1] |> device(ag))

    if ag._current_step < ag._steps_before_learning
        # Only update once we have taken a sufficient number of steps in the replay buffer
        ag._current_step += 1
        return
    end

    for n_update in one(UInt):ag._max_n_updates
        should_update_critic = should_update(ag._q̂_update_ratio, ag._current_step, n_update)
        should_update_actor = should_update(ag._π_update_ratio, ag._current_step, n_update)
        if !should_update_critic && !should_update_actor
            break
        end

        # Sample from replay buffer and reshape if needed
        s_t, a_t, r_tp1, s_tp1, γ_tp1 = rand(
            ag._rng, ag._buffer, ag._batch_size,
        )

        if ag._batch_size == 1
            # Unsqueeze batch dimension
            r_tp1 = [r_tp1]
            γ_tp1 = [γ_tp1]
            s_t = unsqueeze(s_t; dims = ndims(s_t) + 1)
            a_t = unsqueeze(a_t; dims = ndims(a_t) + 1)
            s_tp1 = unsqueeze(s_tp1; dims = ndims(s_tp1) + 1)
        end

        # Policy Evaluation
        if should_update_critic
            ag._current_critic_update += 1
            ag._q̂_update_st, ag._q̂_θ, ag._q̂_st, ag._q̂_target_st, ag._π_st, = update(
                ag._q̂_update_st, ag._π, ag._π_model, ag._π_θ, ag._π_st, ag._q̂,
                ag._q̂_model, ag._q̂_θ, ag._q̂_st, ag._q̂_target_θ, ag._q̂_target_st, s_t, a_t,
                r_tp1, s_tp1, γ_tp1,
            )
        end

        # Policy Improvement
        # Policy improvement should **always** follow policy evaluation in all agents
        if should_update_actor
            ag._current_actor_update += 1
            ag._π_update_st, ag._π_θ, ag._π_st, ag._q̂_st = update(
                ag._π_update_st, ag._π, ag._π_model, ag._π_θ, ag._π_st, ag._q̂, ag._q̂_model,
                ag._q̂_θ, ag._q̂_st, s_t,
            )
        end

        # Update critic target network
        update_target_net = (
            ag._use_target_nets && (
                ag._q̂_target_refresh_steps == 1 ||
                ag._current_critic_update % ag._q̂_target_refresh_steps == 0
            )
        )
        if update_target_net
            polyak!(ag._q̂_polyak_avg, ag._q̂_target_θ, ag._q̂_θ)
        end
    end

    ag._current_step += one(ag._current_step)
    return nothing
end

function stop!(ag::BatchQAgent, r_T, s_T, γ_T)::Nothing
    if !ag._is_training
        @warn "calling stop! on an agent in evaluation mode is a no-op, returning..."
        return nothing
    end

    return nothing
end
