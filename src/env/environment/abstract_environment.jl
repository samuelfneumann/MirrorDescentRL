"""
	AbstractEnvironment

An environment that an agent can interact with.
"""
abstract type AbstractEnvironment end

unwrap(env::AbstractEnvironment) = env
wrapped(env::AbstractEnvironment) = env

"""
    step!(env::AbstractEnvironment, action)

Update the state of `env` based on the dynamics and `action`. Returns the
next state, reward, done flag, and discount factor.

This method should never be overridden for new environments. Instead, override `step!`.
"""
function step!(env::AbstractEnvironment, action)
    if isterminal(env)
        @warn "terminal condition met; call start! before calling step! again"
    end

    return envstep!(env, action)
end

function check_contains_action(env, action)
    if ! contains(action_space(env), action)
        space = action_space(env)
        throw(DomainError(
            action,
            "action $action is not a valid action given action space $space"
        ))
    end
    return action
end

# Interface methods
# The following methods should be implemented for the AbstractEnvironment interface
"""
	start!(env::AbstractEnvironment)::AbstractArray{S}

Reset the environment to some starting state and returns that starting
state.
"""
function start!(env::AbstractEnvironment)::AbstractArray
    return error("start! not implemented for environment $(typeof(env))")
end

"""
    envstep!(env::AbstractEnvironment, action)

Update the state of `env` based on the dynamics and `action`. Returns the
next state, reward, done flag, and discount factor.

This method should be overridden for new environments.
"""
# function envstep!(env::AbstractEnvironment, action)
#     return error("envstep! not implemented for environment $(typeof(env))")
# end

"""
    isterminal(env::AbstractEnvironment)::Bool

Return whether the environment is in a terminal state
"""
function isterminal(env::AbstractEnvironment)::Bool
    return error("isterminal not implemented for environment $(typeof(env))")
end

"""
    γ(env::AbstractEnvironment)

Returns the current discount factor γ. This should be used in place of
env.discount.  """
function γ(env::AbstractEnvironment)
    return error("discount not implemented for environment $(typeof(env))")
end

"""
    observation_space(env::AbstractEnvironment)

Returns the space of observation of the environment
"""
function observation_space(env::AbstractEnvironment)::AbstractSpace
    return error("observation_space not implemented for environment $(typeof(env))")
end

"""
    action_space(env::AbstractEnvironment)

Returns the space of actions of the environment
"""
function action_space(env::AbstractEnvironment)::AbstractSpace
    return error("action_space not implemented for environment $(typeof(env))")
end

"""
    reward(env::AbstractEnvironment)::R where {R<:AbstractFloat}

Return the last reward received
"""
function reward(env::AbstractEnvironment)
    return error("reward not implemented for environment $(typeof(env))")
end

"""
    stop!(::AbstractEnvironment)

Perform cleanup after an experiment is closed, such as freeing up resources.
"""
stop!(::AbstractEnvironment) = nothing
