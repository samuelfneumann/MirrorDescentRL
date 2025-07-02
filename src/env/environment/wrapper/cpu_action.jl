"""
    CPUAction <: AbstractEnvironmentActionWrapper

Moves input actions to the CPU
"""
struct CPUAction{E} <: AbstractEnvironmentActionWrapper where {E<:AbstractEnvironment}
    _env::E
end

action(c::CPUAction, action) = action 
action(c::CPUAction, action::CuArray) = action |> cpu_device()
wrapped(a::CPUAction) = a._env
