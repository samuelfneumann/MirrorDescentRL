"""
    StepLimit <: AbstractEnvironmentWrapper

Limit the steps taken per episode in an environment
"""
mutable struct StepLimit <: AbstractEnvironmentWrapper
    env::AbstractEnvironment
    episode_steps::Integer
    steps_per_episode::Integer

    function StepLimit(env::AbstractEnvironment, steps::Integer)
        return new(env, 0, steps)
    end
end

function StepLimit(env::AbstractEnvironment; steps::Integer)
    StepLimit(env, steps)
end

function start!(t::StepLimit)::AbstractArray
    t.episode_steps = zero(t.episode_steps)
    start!(wrapped(t))
end

function envstep!(t::StepLimit, action)
    t.episode_steps += 1
    state, reward, done, discount = envstep!(wrapped(t), action)

    return state, reward, isterminal(t), discount
end

function reward(t::StepLimit)::AbstractFloat
    return reward(wrapped(t))
end

function action_space(t::StepLimit)::AbstractSpace
    action_space(wrapped(t))
end

function observation_space(t::StepLimit)::AbstractSpace
    observation_space(wrapped(t))
end

function isterminal(t::StepLimit)::Bool
    return isterminal(wrapped(t)) || t.episode_steps >= t.steps_per_episode
end

function γ(t::StepLimit)
    return γ(wrapped(t))
end

function wrapped(t::StepLimit)::AbstractEnvironment
    return t.env
end

function Base.show(io::IO, t::StepLimit)
    print(io, "StepLimit($(t.env); steps=$(t.steps_per_episode))")
end
