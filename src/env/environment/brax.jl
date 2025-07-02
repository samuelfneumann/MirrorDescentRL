# TODO: remove garbage collection from this environment, and instead add it to the
# experiment file.
#
#
# TODO: eventually, I want to move this to PythonCall rather than PyCall, so that this and
# Gymnasium both use the same Python-Julia framework. Further, PythonCall makes code
# nicer, for example converting action and observation spaces.
#
# Adapted from https://github.com/JuliaReinforcementLearning/Brax.jl
module Brax

export BraxEnv

using ..ActorCritic
using Random
using CUDA
using PyCall
using DLPack
using .GC
import Lux: cpu_device

const brax_envs = PyNULL()
const brax_io_html = PyNULL()
const brax_io_image = PyNULL()
const jax = PyNULL()
const dl = PyNULL()

####################################################################
# DLPack: converting jax <-> Julia arrays
####################################################################
from_jax(o) = DLPack.wrap(o, o -> @pycall dl.to_dlpack(o)::PyObject)
to_jax(o) = DLPack.share(o, dl.from_dlpack)

buftype(::CuArray{F,I,B}) where {F,I,B} = B
dldt(::CuArray{F,I,CUDA.Mem.DeviceBuffer}) where {F,I} = DLPack.kDLCUDA
dldt(::CuArray{F,I,CUDA.Mem.HostBuffer}) where {F,I} = DLPack.kDLCUDAHost
dldt(::CuArray{F,I,CUDA.Mem.UnifiedBuffer}) where {F,I} = DLPack.kDLCUDAManaged

# Override broken function in DLPack, see:
# https://github.com/pabloferz/DLPack.jl/issues/33https://github.com/pabloferz/DLPack.jl/issues/33
function DLPack.dldevice(B::CUDA.StridedCuArray)
    return DLPack.DLDevice(dldt(B), CUDA.device(B))
end
####################################################################

mutable struct BraxEnv{B<:Union{Nothing,Int},ST,RF,AS,OS} <: AbstractEnvironment
    const _env::PyObject
    const _step_fn::ST
    const _reset_fn::RF
    _batch_size::B
    _key::PyObject
    const _γ::Float32

    const _action_space::AS
    const _observation_space::OS

    const _garbage_collect_every::Int # steps
    _step::Int
    _collect::Bool

    _state::PyObject

    function BraxEnv(
        name::String;
        γ, seed, garbage_collect_every=25000, unwrap=true, batch_size=nothing, kwargs...,
    )
        name = lowercase(name)
        pyenv = brax_envs.create(name; batch_size=batch_size, kwargs...)

        if unwrap
            pyenv = pyenv.unwrapped
        end

        # Store the jit'd environment step function
        step_fn = jax.jit(pyenv.step)
        SF = typeof(step_fn)

        # Store the jit'd environment reset function
        reset_fn = jax.jit(pyenv.reset)
        RF = typeof(reset_fn)

        # Construct the action space
        high = ones(Float32, pyenv.action_size)
        low = -high
        action_space = Box{Float32}(low, high)
        AS = typeof(action_space)

        # Construct the observation space
        high = [Inf32 for _ in 1:pyenv.observation_size]
        low = [-Inf32 for _ in 1:pyenv.observation_size]
        obs_space = Box{Float32}(low, high)
        OS = typeof(obs_space)

        key = jax.random.PRNGKey(seed)
        env = new{typeof(batch_size),SF,RF,AS,OS}(
            pyenv, step_fn, reset_fn, batch_size, key, γ, action_space, obs_space,
            garbage_collect_every, 1, false,
        )

        # Reset the environment, and populate the _state field
        key1, key2 = jax.random.split(env._key)
        env._key = key1
        env._state = env._reset_fn(key2)

        return env
    end
end

function BraxEnv(
    seed, name::String; γ=0.99f0, batch_size=nothing, kwargs...,
)
    return BraxEnv(name; γ=γ, seed=seed, batch_size=batch_size, kwargs...)
end

function Base.show(io::IO, ::MIME"text/html", env::BraxEnv{Nothing})
    print(io, brax_io_html.render(env._env.sys, [env._state.qp]))
end

function Base.show(io::IO, ::MIME"image/png", env::BraxEnv{Nothing})
    print(io, brax_io_image.render(env._env.sys, [env._state.qp], width=320, height=240))
end

Random.seed!(env::BraxEnv, seed) = env._key = jax.random.PRNGKey(seed)

(env::BraxEnv)(action) = env(to_jax(action))
function (env::BraxEnv)(action::PyObject)
    pycall!(env._state, env._step_fn, PyCall.PyObject, env._state, action)
    return state(env)
end

function ActorCritic.envstep!(env::BraxEnv, action)
    obs = env(action)

    if mod(env._step, env._garbage_collect_every) == 0
        env._collect = true
    end
    env._step += 1

    return obs, reward(env), isterminal(env), γ(env)
end

action_ndims(env::BraxEnv{Nothing}) = env._env.action_size
action_ndims(env::BraxEnv) = env._env.action_size, env._env.batch_size
function ActorCritic.action_space(env::BraxEnv)
    return env._action_space
end

observation_ndims(env::BraxEnv{Nothing}) = env._env.observation_size
observation_ndims(env::BraxEnv) = env._env.observation_size, env._env.batch_size
function ActorCritic.observation_space(env::BraxEnv{Nothing})
    return env._observation_space
end

function ActorCritic.start!(env::BraxEnv)
    if env._collect
        env._collect = false
        # This is needed to ensure that Julia is not holding onto objects which Python no
        # longer needs. Crucially, this **must** be done at episode boundaries.
        #
        # This is likely a problem with Julia 1.9 and can be removed for Julia 1.10
        GC.gc()
    end

    key1, key2 = jax.random.split(env._key)
    env._key = key1
    pycall!(env._state, env._reset_fn, PyCall.PyObject, key2)

    return state(env)
end

state(env::BraxEnv) = env._state.obs |> from_jax

function ActorCritic.reward(env::BraxEnv)
    return (env._state.reward |> from_jax |> cpu_device())[]
end

ActorCritic.isterminal(env::BraxEnv) = env._state.done |> from_jax
function ActorCritic.isterminal(env::BraxEnv{Nothing})
    return (env._state.done |> from_jax |> cpu_device())[] |> Bool
end

ActorCritic.γ(env::BraxEnv) = isterminal(env) ? 0f0 : env._γ

function __init__()
    copy!(brax_envs, pyimport("brax.envs"))
    copy!(brax_io_html, pyimport("brax.io.html"))
    copy!(brax_io_image, pyimport("brax.io.image"))
    copy!(jax, pyimport("jax"))
    copy!(dl, pyimport("jax.dlpack"))

    default_backend = jax.default_backend()
    @info "Brax successfully loaded: jax is using $default_backend as the default backend"
end

end
