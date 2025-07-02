mutable struct OnlineQAgent{Π,ΠM,ΠMΘ,ΠMS,ΠUS,Q,QM,QMΘ,QMS,QUS,RNG} <: AbstractAgent where {
    Π<:AbstractParameterisedPolicy, # π
    ΠM,                             # π Model
    ΠMΘ,                            # π Model Parameters
    ΠMS,                            # π Model State
    ΠUS<:UpdateState,               # π Update State
    Q<:Union{Q,DiscreteQ},          # Value Function
    QM,                             # Value Function Model
    QMΘ,                            # Value Function Model Parameters
    QMS,                            # Value Function Model State
    QUS<:UpdateState,               # Value Function Update State
    RNG<:AbstractRNG,
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
    _q̂_st::QMS
    _q̂_update_st::QUS
    _q̂_update_ratio::UpdateRatio

    _max_n_updates::UInt

    _current_step::UInt
    _rng::RNG
    _is_training::Bool

    function OnlineQAgent(
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
    ) where {P,PM,Q,QM}
        rng = Xoshiro(seed)

        # Initialize π model
        π_θ, π_st = setup(seed, π_model)

        # Initialize q-function model
        q̂_θ, q̂_st = setup(seed, q̂_model)

        # Initialize π update
        π_update_st = setup(
            π_update, env, π, π_model, π_θ, π_st, q̂, q̂_model, q̂_θ, q̂_st, π_optim,
            Lux.replicate(rng),
        )

        # Initialize q-function update
        q̂_update_st = setup(
            q̂_update, π, π_model, π_θ, π_st, q̂, q̂_model, q̂_θ, q̂_st, q̂_optim, seed,
        )

        PMΘ = typeof(π_θ)
        PMS = typeof(π_st)
        PUS = typeof(π_update_st)
        QMΘ = typeof(q̂_θ)
        QMS = typeof(q̂_st)
        QUS = typeof(q̂_update_st)

        π_update_ratio = UpdateRatio(π_update_ratio)
        q̂_update_ratio = UpdateRatio(q̂_update_ratio)

        max_n_updates = max(
            updates_per_step(π_update_ratio)[1], updates_per_step(q̂_update_ratio)[1],
        )
        agent = new{
            P,PM,PMΘ,PMS,PUS,Q,QM,QMΘ,QMS,QUS,typeof(rng),
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
            q̂_st,
            q̂_update_st,
            q̂_update_ratio,
            max_n_updates,
            1,
            rng,
            true,
        )

        # Set the agent to training mode
        train!(agent)
        return agent
    end
end

function train!(agent::OnlineQAgent)::Nothing
    agent._is_training = true
    agent._π_st = train(agent._π_model, agent._π_st)
    agent._q̂_st = train(agent._q̂_model, agent._q̂_st)
    return nothing
end

function eval!(agent::OnlineQAgent)::Nothing
    agent._is_training = false
    agent._π_st = eval(agent._π_model, agent._π_st)
    agent._q̂_st = eval(agent._q̂_model, agent._q̂_st)
    return nothing
end

function select_action(ag::OnlineQAgent{<:AbstractContinuousParameterisedPolicy}, s_t)
    action, ag._π_st = sample(
        ag._π, ag._rng, ag._π_model, ag._π_θ, ag._π_st,
        s_t,
    )
    return action[:, 1]
end

function select_action(ag::OnlineQAgent{<:AbstractDiscreteParameterisedPolicy}, s_t)
    action, ag._π_st = sample(
        ag._π, ag._rng, ag._π_model, ag._π_θ, ag._π_st,
        s_t,
    )
    return action
end

function start!(agent::OnlineQAgent, s_0)::Nothing
    if !agent._is_training
        return
    end
end

function step!(ag::OnlineQAgent, s_t, a_t, r_tp1, s_tp1, γ_tp1)::Nothing
    if !ag._is_training
        return
    end

    # Always using a batch size of 1 -- unsqueeze batch dimensions
    r_tp1 = [r_tp1]
    γ_tp1 = [γ_tp1]
    s_t = reshape(s_t, size(s_t)..., 1)
    # a_t = reshape(a_t, size(a_t)..., 1)
    s_tp1 = reshape(s_tp1, size(s_tp1)..., 1)

    for n_update in one(UInt):ag._max_n_updates
        should_update_critic = should_update(ag._q̂_update_ratio, ag._current_step, n_update)
        should_update_actor = should_update(ag._π_update_ratio, ag._current_step, n_update)
        if !should_update_critic && !should_update_actor
            break
        end

        # Policy Evaluation
        if should_update_critic
            ag._q̂_update_st, ag._q̂_θ, ag._q̂_st, _, ag._π_st = update(
                ag._q̂_update_st, ag._π, ag._π_model, ag._π_θ, ag._π_st, ag._q̂, ag._q̂_model,
                ag._q̂_θ, ag._q̂_st, s_t, a_t, r_tp1, s_tp1, γ_tp1,
            )
        end

        # Policy Improvement
        if should_update_actor
            ag._π_update_st, ag._π_θ, ag._π_st, ag._q̂_st  = update(
                ag._π_update_st, ag._π, ag._π_model, ag._π_θ, ag._π_st, ag._q̂, ag._q̂_model,
                ag._q̂_θ, ag._q̂_st, s_t,
            )
        end
    end

    ag._current_step += one(ag._current_step)
    return nothing
end

function stop!(ag::OnlineQAgent, r_T, s_T, γ_T)::Nothing
    if !ag._is_training
        return nothing
    end

    return nothing
end
