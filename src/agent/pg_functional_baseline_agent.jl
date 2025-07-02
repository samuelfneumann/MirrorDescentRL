"""
"""
mutable struct PGFunctionalBaselineAgent{
    Π,ΠM,ΠMΘ,ΠMS,ΠUS,F,RNG,D,FC,
} <: AbstractAgent where {
    Π<:AbstractParameterisedPolicy, # π
    ΠM,                             # π Model
    ΠMΘ,                            # π Model Parameters
    ΠMS,                            # π Model State
    ΠUS<:UpdateState,               # π Update State
    F<:Function,                    # (G_t, G_0, path_length) -> f(G_t, G_0, path_length)
    RNG<:AbstractRNG,
    D<:AbstractDevice,
    FC<:AbstractFeatureCreator,
}
    # Policy
    _π::Π
    _π_model::ΠM
    _π_θ::ΠMΘ
    _π_st::ΠMS
    _π_update_st::ΠUS

    _baseline_fn::F
    _reward_to_go::Bool

    _buffer::GAEBuffer

    _current_step::UInt
    _current_eps::UInt
    _current_eps_in_epoch::UInt
    _update_every::UInt
    _current_epoch::UInt

    _rng::RNG
    _is_training::Bool

    _feature_creator::FC

    _device::D

    function PGFunctionalBaselineAgent(
        seed::Integer,
        env::AbstractEnvironment,
        π::P,
        π_model::PM,
        π_optim,
        π_update,
        baseline_function::F,
        reward_to_go,
        buffer::GAEBuffer;
        feature_creator::FC=x -> x,
        update_every,
        device,
    ) where {P,PM,F,FC}
        msg = "PGFunctionalBaselineAgent requires GAEBuffer to have λ = 1"
        @assert λ(buffer) == one(λ(buffer)) msg

        rng = if device isa Lux.CUDADevice
            CUDA.RNG(seed)
        elseif device isa Lux.CPUDevice
            rng = Random.default_rng()
            Random.seed!(rng, seed)
        else
            error("unknown device $(typeof(device))")
        end

        # Move the policy to the correct device
        _P = typeof(π)

        # Initialize policy and v function parameters and state
        π_θ, π_st = setup(Lux.replicate(rng), π_model) .|> device

        # Initialize π update
        π_update_st = setup(π_update, π, π_model, π_θ, π_st, π_optim, Lux.replicate(rng))

        PMΘ = typeof(π_θ)
        PMS = typeof(π_st)
        PUS = typeof(π_update_st)

        agent = new{_P,PM,PMΘ,PMS,PUS,F,typeof(rng),typeof(device),FC}(
            π,
            π_model,
            π_θ,
            π_st,
            π_update_st,
            baseline_function,
            reward_to_go,
            buffer,
            1,
            0, # Incremented in start!, so we start from 0
            0, # Incremented in start!, so we start from 0
            update_every,
            1,
            rng,
            true,
            feature_creator,
            device,
        )

        # Set the agent to training mode
        train!(agent)
        return agent
    end
end

device(ag::PGFunctionalBaselineAgent) = ag._device

function train!(agent::PGFunctionalBaselineAgent)::Nothing
    agent._is_training = true
    agent._π_st = train(agent._π_model, agent._π_st)
    return nothing
end

function eval!(agent::PGFunctionalBaselineAgent)::Nothing
    agent._is_training = false
    agent._π_st = eval(agent._π_model, agent._π_st)
    return nothing
end

function select_action(
    ag::PGFunctionalBaselineAgent{<:AbstractContinuousParameterisedPolicy}, s_t,
)
    s_t = s_t |> device(ag) |> ag._feature_creator
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
    ag::PGFunctionalBaselineAgent{<:AbstractDiscreteParameterisedPolicy}, s_t::AbstractVector,
)
    s_t = s_t |> device(ag) |> ag._feature_creator
    if !ag._is_training
        error("not implemented")
    else
        action, ag._π_st = sample(
            ag._π, ag._rng, ag._π_model, ag._π_θ, ag._π_st, s_t,
        )
        return action
    end
end

function start!(agent::PGFunctionalBaselineAgent, s_0)::Nothing
    if !agent._is_training
        @warn "calling start! on an agent in evaluation mode is a no-op, returning..."
        return
    end

    agent._current_eps += 1
    agent._current_eps_in_epoch += 1

    return nothing
end

function step!(ag::PGFunctionalBaselineAgent, s_t, a_t, r_tp1, _s_tp1, γ_tp1)::Nothing
    ag._current_step += 1

    if !ag._is_training
        @warn "calling step! on an ag in evaluation mode is a no-op, returning..."
        return
    end

    s_t = s_t |> device(ag)
    a_t = a_t |> device(ag)

    # Add transition to the replay buffer
    if !full(ag._buffer)
        push!(ag._buffer, s_t, a_t, r_tp1 |> device(ag), γ_tp1 |> device(ag), 0f0)
    else
        @warn "buffer is full: transitions will not be added to the buffer util it is reset"
    end

    return nothing
end

function _step!(ag::PGFunctionalBaselineAgent)
    # Sample from GAEBuffer. At this point A_t == G_t == reward to go estimate
    s_t, a_t, path_lengths, G_0, G_t, A_t, γ_tp1 = get(ag._buffer)

    s_t = ag._feature_creator(s_t)

    # Calculate the gradient scale
    scale = if ag._reward_to_go
        A_t
    else
        [G_0[i] for i in eachindex(G_0) for _ in 1:path_lengths[i]]
    end

    # Adjust scale for baseline
    baseline = ag._baseline_fn(G_t, G_0)
    scale .-= baseline

    # Policy Improvement
    ag._π_update_st, ag._π_θ, ag._π_st = update(
        ag._π_update_st, ag._π, ag._π_model, ag._π_θ, ag._π_st, s_t, a_t, scale, γ_tp1,
    )

    reset!(ag._buffer)
    ag._current_epoch += 1

    return nothing
end

function stop!(ag::PGFunctionalBaselineAgent, r_T, s_T, γ_T)::Nothing
    if !ag._is_training
        @warn "calling stop! on an agent in evaluation mode is a no-op, returning..."
        return nothing
    end

    finish_path!(ag._buffer, 0f0)

    if mod(ag._current_eps, ag._update_every) == 0
        _step!(ag)
        ag._current_eps_in_epoch = 0
    end

    return nothing
end
