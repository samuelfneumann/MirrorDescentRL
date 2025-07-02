"""
    ClipAction <: AbstractEnvironmentActionWrapper

Clips the continuous actions to within the action bounds of `E`.
"""
struct ClipAction{E} <: AbstractEnvironmentActionWrapper where {E<:AbstractEnvironment}
    _env::E

    function ClipAction(env::E) where {E<:AbstractEnvironment}
        if !continuous(action_space(env))
            error("must use continuous action environments with ClipAction")
        end

        return new{E}(env)
    end
end

function action(c::ClipAction, action)
    lower = low(action_space(wrapped(c)))
    upper = high(action_space(wrapped(c)))

    return clamp.(action, lower, upper)
end

function action(c::ClipAction, action::CuArray)
    lower = low(action_space(wrapped(c))) |> gpu_device()
    upper = high(action_space(wrapped(c))) |> gpu_device()

    return clamp.(action, lower, upper)
end

function wrapped(a::ClipAction{E})::E where {E<:AbstractEnvironment}
    return a._env
end
