mutable struct PGAgent{
    Π,ΠM,ΠMΘ,ΠMS,ΠUS,V,VM,VMΘ,VMS,VUS,RNG,D,PFC,VFC,
} <: AbstractAgent where {
    Π<:AbstractParameterisedPolicy, # π
    ΠM,                             # π Model
    ΠMΘ,                            # π Model Parameters
    ΠMS,                            # π Model State
    ΠUS<:UpdateState,               # π Update State
    V<:AbstractStateValueFunction,
    VM,                                                     # Value Function Model
    VMΘ,                                                    # Value Function Model Parameters
    VMS,                                                    # Value Function Model State
    VUS<:UpdateState{<:AbstractStateValueCriticUpdate},     # Value Function Update State
    RNG<:AbstractRNG,
    D<:AbstractDevice,
    PFC<:Union{Nothing,AbstractFeatureCreator},
    VFC<:Union{Nothing,AbstractFeatureCreator},
}
    # Policy
    _π::Π
    _π_model::ΠM
    _π_θ::ΠMΘ
    _π_st::ΠMS
    _π_update_st::ΠUS

    # V function
    _v̂::V
    _v̂_model::VM
    _v̂_θ::VMΘ
    _v̂_st::VMS
    _v̂_update_st::VUS

    _buffer::GAEBuffer

    _current_step::UInt
    _current_eps::UInt
    _current_eps_in_epoch::UInt
    _update_every::UInt
    _n_epochs::UInt

    _rng::RNG
    _is_training::Bool
    _last_path_finished::Bool

    _p_feature_creator::PFC
    _v_feature_creator::VFC

    _device::D

    function PGAgent(
        seed::Integer,
        env::AbstractEnvironment,
        π::P,
        π_model::PM,
        π_optim,
        π_update,
        v̂::V,
        v̂_model::VM,
        v̂_optim,
        v̂_update::AbstractStateValueCriticUpdate,
        buffer::GAEBuffer;
        policy_feature_creator::PFC=nothing,
        value_function_feature_creator::VFC=nothing,
        update_every,
        device,
    ) where {P,PM,V,VM,PFC,VFC}
        rng = if device isa Lux.CUDADevice
            CUDA.RNG(seed)
        elseif device isa Lux.CPUDevice
            rng = Random.default_rng()
            Random.seed!(rng, seed)
        else
            error("unknown device $(typeof(device))")
        end

        # Move the policy to the correct device
        π = π |> device
        _P = typeof(π)

        # Initialize policy and v function parameters and state
        π_θ, π_st = setup(Lux.replicate(rng), π_model) .|> device
        v̂_θ, v̂_st = setup(Lux.replicate(rng), v̂_model) .|> device

        # Initialize π update
        π_update_st = setup(π_update, π, π_model, π_θ, π_st, π_optim, Lux.replicate(rng))

        # Initialize v-function update
        v̂_update_st = setup(
            v̂_update, π, π_model, π_θ, π_st, v̂, v̂_model, v̂_θ, v̂_st, v̂_optim,
            Lux.replicate(rng),
        )

        buffer = buffer |> device

        PMΘ = typeof(π_θ)
        PMS = typeof(π_st)
        PUS = typeof(π_update_st)
        VMΘ = typeof(v̂_θ)
        VMS = typeof(v̂_st)
        VUS = typeof(v̂_update_st)

        agent = new{
            _P,PM,PMΘ,PMS,PUS,V,VM,VMΘ,VMS,VUS,typeof(rng),typeof(device),PFC,VFC,
        }(
            π,
            π_model,
            π_θ,
            π_st,
            π_update_st,
            v̂,
            v̂_model,
            v̂_θ,
            v̂_st,
            v̂_update_st,
            buffer,
            0,
            0,
            0,
            update_every,
            0,
            rng,
            true,
            false,
            policy_feature_creator,
            value_function_feature_creator,
            device,
        )

        # Set the agent to training mode
        train!(agent)
        return agent
    end
end

device(ag::PGAgent) = ag._device

function train!(agent::PGAgent)::Nothing
    agent._is_training = true
    agent._π_st = train(agent._π_model, agent._π_st)
    agent._v̂_st = train(agent._v̂_model, agent._v̂_st)
    return nothing
end

function eval!(agent::PGAgent)::Nothing
    agent._is_training = false
    agent._π_st = eval(agent._π_model, agent._π_st)
    agent._v̂_st = eval(agent._v̂_model, agent._v̂_st)
    return nothing
end

function select_action(ag::PGAgent{<:AbstractContinuousParameterisedPolicy}, s_t)
    s_t = s_t |> device(ag)
    if !ag._is_training
        error("not implemented")
    else
        action, ag._π_st = sample(
            ag._π, ag._rng, ag._π_model, ag._π_θ, ag._π_st, s_t,
        )

        return action[:, 1]
    end
end

function select_action(
    ag::PGAgent{<:AbstractDiscreteParameterisedPolicy}, s_t::AbstractVector,
)
    s_t = s_t |> device(ag)
    if !ag._is_training
        error("not implemented")
    else
        action, ag._π_st = sample(
            ag._π, ag._rng, ag._π_model, ag._π_θ, ag._π_st, s_t,
        )

        return action
    end
end

function start!(agent::PGAgent, s_0)::Nothing
    if !agent._is_training
        @warn "calling start! on an agent in evaluation mode is a no-op, returning..."
        return
    end

    agent._current_eps += 1
    agent._current_eps_in_epoch += 1

    return nothing
end

function step!(ag::PGAgent, s_t, a_t, r_tp1, _s_tp1, γ_tp1)::Nothing
    ag._current_step += 1

    if !ag._is_training
        @warn "calling step! on an ag in evaluation mode is a no-op, returning..."
        return
    end

    s_t = s_t |> device(ag)
    a_t = a_t |> device(ag)

    v_t, ag._v̂_st = predict(ag._v̂, ag._v̂_model, ag._v̂_θ, ag._v̂_st, s_t)

    # Add transition to the replay buffer
    if !full(ag._buffer)
        push!(ag._buffer, s_t, a_t, r_tp1 |> device(ag), γ_tp1 |> device(ag), v_t)
    else
        @warn "buffer is full: transitions will not be added to the buffer util it is reset"
    end

    if γ_tp1 ≈ zero(γ_tp1)
        finish_path!(ag._buffer, 0)
        ag._last_path_finished = true
    end

    return nothing
end

function _step!(ag::PGAgent)
    # Sample from GAEBuffer
    s_t, a_t, _, _, G_t, A_t, γ_tp1 = get(ag._buffer)

    # Policy Evaluation
    ag._v̂_update_st, ag._v̂_θ, ag._v̂_st = update(
        ag._v̂_update_st, ag._v̂, ag._v̂_model, ag._v̂_θ, ag._v̂_st, s_t, G_t,
    )

    # Policy Improvement
    ag._π_update_st, ag._π_θ, ag._π_st = update(
        ag._π_update_st, ag._π, ag._π_model, ag._π_θ, ag._π_st, s_t, a_t, A_t, γ_tp1,
    )

    reset!(ag._buffer)
    ag._n_epochs += 1

    return nothing
end

function stop!(ag::PGAgent, r_T, s_T, γ_T)::Nothing
    if !ag._is_training
        @warn "calling stop! on an agent in evaluation mode is a no-op, returning..."
        return nothing
    end

    if !agent._last_path_finished
        v_t, ag._v̂_st = predict(ag._v̂, ag._v̂_model, ag._v̂_θ, ag._v̂_st, s_T)
        finish_path!(agent._buffer, v_t)
    end
    agent._last_path_finished = false

    if mod(ag._current_eps, ag._update_every) == 0
        _step!(ag)
        ag._current_eps_in_epoch = 0
    end

    return nothing
end
