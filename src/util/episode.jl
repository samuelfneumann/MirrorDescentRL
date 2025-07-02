# Taken from MinimalRLCore.jl
# https://github.com/mkschleg/MinimalRLCore.jl/blob/master/src/episode.jl
export AbstractEpisode, Episode, run!

using Random

"""
    AbstractEpisode

# Interface
- `_start!`: This returns the initial state of the episode iteration and starts the
    environment and agent.
- `_step!`:
- `_stop!`:
- `_done`:
- `total_reward`: Return the total cummulative reward for an episode.
"""
abstract type AbstractEpisode end

function Base.iterate(ep::AbstractEpisode, state = _start!(ep))
    s_t = state[1][2]

    if _done(ep, state)
        r_t = state[1][1]
        γ_t = state[1][3]
        _stop!(ep, r_t, s_t, γ_t)
        return nothing
    end

    a_t, s_tp1, r_tp1, t, γ_tp1, agent_ret = if state[2] == 1
        _step!(ep, s_t, ep.initial_action)
    else
        _step!(ep, s_t)
    end

    return (
        (s_t, a_t, r_tp1, s_tp1, t, γ_tp1, agent_ret),
        ((r_tp1, s_tp1, γ_tp1, agent_ret), state[2] + 1),
    )
end

function run!(ep::AbstractEpisode)
    num_steps = 0
    for sarsγ ∈ ep
        num_steps += 1
    end
    total_reward(ep), num_steps
end

function run!(f::Base.Callable, ep::AbstractEpisode)
    num_steps = 0
    for sarsγ ∈ ep
        f(sarsγ)
        num_steps += 1
    end
    total_reward(ep), num_steps
end

function run!(
    f::Base.Callable, env::AbstractEnvironment, agent::AbstractAgent, args...; kwargs...
)
    run!(f, Episode(env, agent, args...; kwargs...))
end

function run!(env::AbstractEnvironment, agent::AbstractAgent, args...; kwargs...)
    run!(Episode(env, agent, args...; kwargs...))
end


"""
    Episode(env, agent, rng)

This is a struct for managing the components of an episode iterator. You should only pass a
reference to env and agent, while managing the reference separately.

# Arguments
- `env::AbstractEnvironment`: The Environment (following RLCore interface)
- `agent::AbstractAgent`: The Agent (following RLCore interface)

"""
mutable struct Episode{
    E<:AbstractEnvironment,
    A<:AbstractAgent,
    F<:Number,
    ACT<:Union{AbstractArray,Nothing},
} <: AbstractEpisode
    env::E
    agent::A
    total_reward::F # total reward of the episode

    max_steps::Int
    curr_step::Int

    initial_action::ACT
end

function Episode(env, agent, steps; initial_action=nothing)
    Episode(env, agent, zero(reward(env)), steps, 0, initial_action)
end

total_reward(ep::Episode) = ep.total_reward
_done(ep::Episode, state) = isterminal(ep.env) || ep.curr_step >= ep.max_steps

function _start!(ep::Episode)
    s = start!(ep.env)
    agent_ret = start!(ep.agent, s)
    ep.total_reward = zero(ep.total_reward)
    return (nothing, s, nothing, agent_ret), 1
end

function _step!(ep::Episode, s_t)
    a_t = select_action(ep.agent, s_t)
    return _step!(ep, s_t, a_t)
end

_step!(ep::Episode, s_t, ::Nothing) = _step!(ep, s_t)

function _step!(ep::Episode, s_t, a_t)
    ep.curr_step += 1

    s_tp1, r_tp1, t, γ_tp1 = step!(ep.env, a_t)
    agent_ret = step!(ep.agent, s_t, a_t, r_tp1, s_tp1, γ_tp1)

    ep.total_reward += r_tp1

    return a_t, s_tp1, r_tp1, t, γ_tp1, agent_ret
end

function _stop!(ep::Episode, r_T, s_T, γ_T)
    stop!(ep.agent, r_T, s_T, γ_T)
end
