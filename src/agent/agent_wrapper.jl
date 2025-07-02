abstract type AbstractAgentWrapper <: AbstractAgent end

function wrapped(t::AbstractAgentWrapper)
    error("wrapped not implemented for type $(typeof(t))")
end

function unwrap(t::AbstractAgentWrapper)
    error("unwrap not implemented for type $(typeof(t))")
end

device(ag::AbstractAgentWrapper) = device(wrapped(ag))
train!(ag::AbstractAgentWrapper) = train!(wrapped(ag))
eval!(ag::AbstractAgentWrapper) = eval!(wrapped(ag))
select_action(ag::AbstractAgentWrapper, s_t) = select_action(wrapped(ag), s_t)
start!(ag::AbstractAgentWrapper, s_0) = start!(wrapped(ag), s_0)
stop!(ag::AbstractAgentWrapper, r_T, s_T, γ_T) = stop!(wrapped(ag), r_T, s_T, γ_T)

function step!(ag::AbstractAgentWrapper, s_t, a_t, r_tp1, s_tp1, γ_tp1)
    step!(wrapped(ag), s_t, a_t, r_tp1, s_tp1, γ_tp1)
end


abstract type AbstractAgentActionWrapper <: AbstractAgentWrapper end

function action(ag::AbstractAgentActionWrapper, a)
    error("action not implemented for type $(typeof(ag))")
end

function select_action(ag::AbstractAgentActionWrapper, s_t)
    a = select_action(wrapped(ag), s_t)
    return action(ag, a)
end


mutable struct RandomFirstActionAgent{
    A<:AbstractAgent,S,R<:AbstractRNG,
} <: AbstractAgentActionWrapper
    const _agent::A
    _first_action::Bool
    const _action_space::S
    const _rng::R

    function RandomFirstActionAgent(
        ag::A, env::AbstractEnvironment, rng::R,
    ) where {A,R<:AbstractRNG}
        as = action_space(env)
        S = typeof(as)
        return new{A,S,R}(ag, true, as, deepcopy(rng))
    end
end

function action(ag::RandomFirstActionAgent, a)
    return if ag._first_action
        ag._first_action = false
        rand(ag._rng, ag._action_space)
    else
        a
    end
end

wrapped(ag::RandomFirstActionAgent) = ag._agent
unwrap(ag::RandomFirstActionAgent) = unwrap(wrapped(ag))

function stop!(ag::RandomFirstActionAgent, r_T, s_T, γ_T)::Nothing
    ag._first_action = true
    return nothing
end
