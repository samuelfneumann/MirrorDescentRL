# Adapted from https://github.com/mkschleg/Gymnasium.jl/tree/main

module Gymnasium

using ..ActorCritic
using .GC

export GymnasiumEnv

import PythonCall: PythonCall, Py, @py, pyconvert, pybuiltins, pyimport, pyis

pygym::Py = PythonCall.pynew() # initially NULL
sys::Py = PythonCall.pynew()
tbu::Py = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(sys, pyimport("sys"))
    sys.path.insert(0, ".")

    PythonCall.pycopy!(tbu, pyimport("truck_backer_upper.trucker_backer_env"))

    PythonCall.pycopy!(pygym, pyimport("gymnasium"))
    PythonCall.pyconvert_add_rule(
        "gymnasium.spaces.discrete:Discrete", Discrete, convert_discrete_space,
    )

    PythonCall.pyconvert_add_rule("gymnasium.spaces.box:Box", Box, convert_box_space)
    @info "Gymnasium successfully loaded"
end


######## Spaces #########
function convert_discrete_space(::Type{Discrete}, pyobj::Py)
    n = pyconvert(Int, pyobj.n)
    start = pyconvert(Int, pyobj.start)
    if start != 0
        return pyconvert_unconverted()
    end
    ds = Discrete((1,), n)
    return Gymnasium.PythonCall.pyconvert_return(ds)
end

function convert_box_space(::Type{Box}, pyobj::Py)
    low = pyconvert(Vector{Float32}, pyobj.low)
    high = pyconvert(Vector{Float32}, pyobj.high)
    box = Box(low, high)
    return PythonCall.pyconvert_return(box)
end
################

######## Environment ########
mutable struct GymnasiumEnv{T,AS,OS} <: AbstractEnvironment
    pyenv::Py
    const id::String
    const action_space::AS
    const observation_space::OS
    const γ::Float32

    observation::T
    terminal::Bool
    reward::Float32
    info::Py

    _garbage_collect_every::Int
    _steps::Int
end

function GymnasiumEnv(id::String, pyenv::Py; seed, γ, garbage_collect_every=25000)
    obs, info = pyenv.reset(seed=seed) # reset to get the obs type
    as = pyconvert(Any, pyenv.action_space)
    os = pyconvert(Any, pyenv.observation_space)
    env = GymnasiumEnv(
        pyenv, id, as, os, Float32(γ), convert_obs(obs), false, 0.0f0, info,
        garbage_collect_every, 1,
    )
    return env
end

GymnasiumEnv(id::String; kwargs...) = make(id; kwargs...)
GymnasiumEnv(name::String, version::Int; kwargs...) = make(name, version; kwargs...)

ActorCritic.action_space(g::GymnasiumEnv) = g.action_space
ActorCritic.observation_space(g::GymnasiumEnv) = g.observation_space
ActorCritic.reward(g::GymnasiumEnv) = g.reward
ActorCritic.isterminal(g::GymnasiumEnv) = g.terminal
ActorCritic.γ(g::GymnasiumEnv) = isterminal(g) ? 0f0 : g.γ

ispy(::GymnasiumEnv) = true
Py(env::GymnasiumEnv) = env.pyenv

function convert_obs(::GymnasiumEnv{T}, pyobs::Py) where T
    pyconvert(T, pyobs)
end

function convert_obs(pyobs::Py)
    t_str = pyconvert(String, @py type(pyobs).__name__)
    convert_obs(Val(Symbol(t_str)), pyobs)
end

convert_obs(::Val{:ndarray}, pyobs::Py) = pyconvert(Vector{Float32}, pyobs)

function make(
    id::String; seed, γ, unwrap=true, render_mode=nothing, max_episode_steps=nothing,
    autoreset=false, disable_env_checker=nothing, kwargs...,
)
    display(kwargs)
    pyenv = if lowercase(id) ∈ ["truck_backer_upper", "truckbackerupper"]
        @assert unwrap
        tbu.TruckBackerEnv()
    else
        pygym.make(
            id, render_mode=render_mode, max_episode_steps=max_episode_steps,
            autoreset=autoreset, disable_env_checker=disable_env_checker, kwargs...,
        )

        if unwrap
            pyenv = pyenv.unwrapped
        end
    end

    return GymnasiumEnv(id, pyenv; seed=seed, γ=γ)
end

function make(name::String, version::Int; kwargs...)
    id = name * "-v" * string(version)
    make(id::String; kwargs...)
end

convert_action(::GymnasiumEnv, action) = action
function convert_action(::GymnasiumEnv{T,AS,Discrete}, action) where {T,AS}
    return only(action) - 1
end

function ActorCritic.envstep!(env::GymnasiumEnv, action)
    action = convert_action(env, action)

    if pyis(env.pyenv, pybuiltins.None)
        throw("GymnasiumEnv: pyenv None in envstep!")
    end

    obs, rew, terminal, _, info = env.pyenv.step(action)
    env.info = info
    env.observation = convert_obs(env, obs)
    env.terminal = pyconvert(Bool, terminal)
    env.reward = pyconvert(Float32, rew)

    if mod(env._steps, env._garbage_collect_every) == 0
        # This is needed to ensure that Julia is not holding onto objects which Python no
        # longer needs
        GC.gc()
    end
    env._steps += 1

    return env.observation, reward(env), isterminal(env), γ(env)
end

function ActorCritic.start!(
    env::GymnasiumEnv{T};
    seed::Union{Nothing,Int}=nothing, options::Union{Nothing,Dict}=nothing,
) where {T}
    if pyis(env.pyenv, pybuiltins.None)
        throw("GymnasiumEnv: pyenv None in start!")
    end

    obs, info = env.pyenv.reset(seed=seed, options=options)
    env.observation = pyconvert(T, obs)
    env.info = info
    env.terminal = false

    return env.observation
end

function ActorCritic.render(env::GymnasiumEnv)
    if pyis(env.pyenv, pybuiltins.None)
        throw("GymnasiumEnv: pyenv None in render")
    end
    env.pyenv.render()
    return nothing
end


function ActorCritic.stop!(env::GymnasiumEnv)
    if pyis(env.pyenv, pybuiltins.None)
        return
    end
    env.pyenv.close()
    env.pyenv = pybuiltins.None
    return nothing
end

end
