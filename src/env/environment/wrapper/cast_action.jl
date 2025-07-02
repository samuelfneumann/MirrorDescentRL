"""
    CastAction <: AbstractEnvironmentActionWrapper

Casts an action to be the appropriate type for an environment
"""
struct CastAction{E} <: AbstractEnvironmentActionWrapper
    _env::E

    function CastAction(env::E) where {E<:AbstractEnvironment}
        return new{E}(env)
    end
end

function action(a::CastAction, action::AbstractArray)
    return convert.(eltype(action_space(a._env)), action)
end

function wrapped(a::CastAction{E})::E where {E}
	return a._env
end
