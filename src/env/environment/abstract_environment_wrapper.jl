abstract type AbstractEnvironmentWrapper <: AbstractEnvironment end

function wrapped(env::AbstractEnvironmentWrapper)::AbstractEnvironment
    return error("wrapped not implemented for AbstractEnvironmentWrapper $(typeof(env))")
end

function unwrap(env::AbstractEnvironmentWrapper)::AbstractEnvironment
    return unwrap(wrapped(env))
end

function unwrap_all(env::AbstractEnvironmentWrapper)::Vector{AbstractEnvironment}
    return [env, unwrap_all(wrapped(env))...]
end

function unwrap_all(env::AbstractEnvironment)::Vector{AbstractEnvironment}
    return [env]
end

function action_space(env::AbstractEnvironmentWrapper)::AbstractSpace
    return action_space(wrapped(env))
end

function observation_space(env::AbstractEnvironmentWrapper)::AbstractSpace
    return observation_space(wrapped(env))
end

info(env::AbstractEnvironmentWrapper)::Dict{String,Any} = Dict{String, Any}()

function reward(env::AbstractEnvironmentWrapper)
    return reward(wrapped(env))
end
function γ(env::AbstractEnvironmentWrapper)
    return γ(wrapped(env))
end

function isterminal(env::AbstractEnvironmentWrapper)::Bool
    return isterminal(wrapped(env))
end

function start!(env::AbstractEnvironmentWrapper)
    return start!(wrapped(env))
end

function envstep!(env::AbstractEnvironmentWrapper, action)
    return envstep!(wrapped(env), action)
end

# ###################################
# Action wrapper
# ###################################
"""
    AbstractEnvironmentActionWrapper <: AbstractEnvironmentWrapper

Wrap an environment to adjust its actions

## Interface
In order to satisfy this interface and sub-type an AbstractEnvironmentActionWrapper, you
must implement the following functions:

    action(a::AbstractEnvironmentActionWrapper, action)
    wrapped(env::AbstractEnvironmentActionWrapper)::AbstractEnvironment
"""
abstract type AbstractEnvironmentActionWrapper <: AbstractEnvironmentWrapper end

"""
    action(a::AbstractEnvironmentActionWrapper, action)

Adjust the action `action` based on the action wrapper `a`.
"""
function action(a::AbstractEnvironmentActionWrapper, action)
    return error("action not implemented for AbstractEnvironmentActionWrapper $(typeof(env))")
end

function envstep!(wrapper::AbstractEnvironmentActionWrapper, act)
    return envstep!(wrapped(wrapper), action(wrapper, act))
end

function start!(wrapper::AbstractEnvironmentActionWrapper)
    return start!(wrapped(wrapper))
end

function step!(env::AbstractEnvironmentActionWrapper, act)
    if isterminal(env)
        @warn "terminal condition met; call start! before calling step! again"
    end

    return envstep!(env, act)
end

# ###################################
# Reward wrapper
# ###################################
"""
    AbstractEnvironmentRewardWrapper <: AbstractEnvironmentWrapper end

Alter the reward returned by an environment.

To implement this interface, a type should implement both `reward` methods and the
`set_last_reward!` method.
"""
abstract type AbstractEnvironmentRewardWrapper <: AbstractEnvironmentWrapper end

"""
    reward(a::AbstractEnvironmentRewardWrapper, r::AbstractFloat)::AbstractFloat
    reward(a::AbstractEnvironmentRewardWrapper)::AbstractFloat

Alter some reward `r`.
"""
function reward(
    a::AbstractEnvironmentRewardWrapper,
    r::Real,
)::AbstractFloat
    error("reward not implemented by AbstractEnvironmentRewardWrapper $(typeof(a))")
end

function reward(a::AbstractEnvironmentRewardWrapper)::AbstractFloat
    error("reward not implemented by AbstractEnvironmentRewardWrapper $(typeof(a))")
end

"""
    set_last_reward!(a::AbstractEnvironmentWrapper, r::AbstractFloat)

Sets the last reward seen in the environment to be `r`.

This function should not change the value of `r` before setting.
"""
function set_last_reward!(a::AbstractEnvironmentWrapper, r::Real)
    error(
        "set_last_reward! not implemented by AbstractEnvironmentRewardWrapper $(typeof(a))",
    )
end

function envstep!(a::AbstractEnvironmentRewardWrapper, action)
    obs, r, done, γ = envstep!(wrapped(a), action)
    r = reward(a, r)
    set_last_reward!(a, r)

    return obs, r, done, γ
end

# ###################################
# Observation wrapper
# ###################################

"""
    AbstractEnvironmentObservationWrapper <: AbstractEnvironmentWrapper end

Alter an observation from some environment.
"""
abstract type AbstractEnvironmentObservationWrapper <: AbstractEnvironmentWrapper end

"""
    observation(
        o::AbstractEnvironmentObservationWrapper,
        obs::AbstractArray,
    )::AbstractArray

Alter the observation `obs`.

This function must be implemented by any `ObservationWrapper`
"""
function observation(
    o::AbstractEnvironmentObservationWrapper,
    obs,
)::AbstractArray
    error(
        "obsevation not implemented by AbstractEnvironmentObservationWrapper $(typeof(o))",
    )
end

function start!(o::AbstractEnvironmentObservationWrapper)::AbstractArray
    return observation(o, start!(wrapped(o)))
end

function envstep!(o::AbstractEnvironmentObservationWrapper, action)
    obs, r, done, γ = envstep!(wrapped(o), action)

    return observation(o, obs), r, done, γ
end

function observation_space(t::AbstractEnvironmentObservationWrapper)
    error("observation_space not implemented for type $(typeof(t))")
end
