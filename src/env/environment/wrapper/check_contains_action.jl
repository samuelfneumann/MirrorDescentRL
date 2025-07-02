"""
    CheckContainsAction <: AbstractEnvironmentActionWrapper

Checks if an action space contains an action
"""
struct CheckContainsAction{E} <: AbstractEnvironmentActionWrapper where {E<:AbstractEnvironment}
    _env::E
end

action(c::CheckContainsAction, action) = check_contains_action(c._env, action)
function action(c::CheckContainsAction, action::CuArray)
    check_contains_action(c._env, action |> cpu_device())
    return action
end
wrapped(a::CheckContainsAction) = a._env
