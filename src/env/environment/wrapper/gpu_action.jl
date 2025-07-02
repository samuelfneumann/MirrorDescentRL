"""
    GPUAction <: AbstractEnvironmentActionWrapper

Moves input actions to the GPU
"""
struct GPUAction{E} <: AbstractEnvironmentActionWrapper where {E<:AbstractEnvironment}
    _env::E
end

action(c::GPUAction, action) = action 
action(c::GPUAction, action::CuArray) = action |> gpu_device()
wrapped(a::GPUAction) = a._env
