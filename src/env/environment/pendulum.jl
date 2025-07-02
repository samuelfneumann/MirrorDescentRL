"""
    Pendulum <: AbstractEnvironment

# Description
The inverted pendulum swingup problem is based on the classic problem in
control theory. The system consists of a pendulum attached at one end to a
fixed point, and the other end being free. The pendulum starts facing
straight downwards, an angle of ±π. The goal is to apply torque on the
free end to swing it into an upright position, an angle of 0, with its
centre of gravity right above the fixed point.

## Observation Space
If `trig_features == True`, then the observation is an `ndarray` with shape
`(3,)` representing the x-y coordinates of the pendulum's free end and its
angular velocity:
| Num | Observation  | Min  | Max |
|-----|--------------|------|-----|
| 0   | x = cos(θ)   | -1.0 | 1.0 |
| 1   | y = sin(θ)   | -1.0 | 1.0 |
| 2   | θ̇            | -8.0 | 8.0 |

If `trig_features == False`, then the observation is an `ndarray` with
shape `(2,)` representing the pendulum's angle and angular velocity:
| Num | Observation | Min  | Max |
|-----|-------------|------|-----|
| 0   | θ           | -π   | π   |
| 1   | θ̇           | -8.0 | 8.0 |

The angle of the pendulum is θ and is normalized/wrapped to stay
in [-π, π).

## Action Space
In the continuous-action setting, the action is a `ndarray` with shape
`(1,)` representing the torque applied to free end of the pendulum.
| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

In the discrete-action setting, the action is an integer in the set
`{0, 1, 2}`, corresponding to maximum torque in the negative direction,
no torque, maximum torque in the positive direction.

## Rewards
The reward function is defined as cos(θ) when `sparse_rewards == False`.
Otherwise, the reward is +1 if |θ| < π/12 and 0 otherwise.

## Starting State
The starting state is facing straight down with no velcoity: [-π, 0].

## Episode termination
This environment is continuing and does not have a termination.
"""
mutable struct Pendulum{
        A<:AbstractSpace,
        O<:AbstractSpace,
        R<:AbstractRNG,
} <: AbstractEnvironment
    const _observationspace::O
    const _actionspace::A
    const _γ::Float32
    const _sparse_rewards::Bool

    const _max_speed::Float32
    const _max_torque::Float32
    const _dt::Float32

    const _mass::Float32
    const _length::Float32
    const _gravity::Float32

    _rng::R

    _state::Vector{Float32}
    const _trig_features::Bool

    function Pendulum(
        rng::R,
        continuous,
        sparse_rewards,
        trig_features,
        γ,
    ) where {R<:AbstractRNG}
        max_speed = 8f0
        max_torque = 2f0
        dt = 0.05f0
        mass = 1f0
        length = 1f0
        gravity = 10f0

        if !trig_features
            # Encode states as [θ, θ̇]
            high = Float32[π, max_speed]
            low = -high
        else
            # Encode states as [cos(θ), sin(θ), θ̇]
            high = Float32[1f0, 1f0, max_speed]
            low = -high
        end

        obs_space = Box{Float32}(low, high)
        action_space = continuous ? Box{Float32}(-max_torque, max_torque) : Discrete(3)
        O = typeof(obs_space)
        A = typeof(action_space)

        p = new{A,O,R}(
            obs_space,
            action_space,
            γ,
            sparse_rewards,
            max_speed,
            max_torque,
            dt,
            mass,
            length,
            gravity,
            rng,
            Float32[-π , 0f0],
            trig_features,
        )

        start!(p)
        return p
    end
end

function Pendulum(
    rng::AbstractRNG;
    continuous,
    γ=0.99f0,
    trig_features = false,
    sparse_rewards = false,
)
    return Pendulum(rng, continuous, sparse_rewards, trig_features, γ)
end

function start!(p::Pendulum)
    return p._state = Float32[-π, 0f0]
end

function envstep!(p::Pendulum, action)
    check_contains_action(p, action)

    # Sanitize the input action depending on whether we are in the
    # discrete or continuous setting
    if !(p |> action_space |> continuous)
        u = _discrete_action(p, action)
    else
        u = _continuous_action(p, action)
    end

    θ, dθ = p._state
    g = p._gravity
    m = p._mass
    l = p._length
    new_dθ = dθ + (-3f0 * g / (2f0 * l) * sin(θ + π) + 3f0 / (m * l^2) * u) * p._dt

    # The next two lines should probably be switched. Right now, we use an unclipped
    # new_dθ to calculate the new angle new_θ, but when calculating the new_dθ, we use the
    # previously clipped old dθ. This is okay. I'm keeping it like this for now because
    # all of my current results on Pendulum use this version. This problematic
    # implementation was taken from OpenAI's gym package, see:
    #
    # https://github.com/openai/gym/blob/78d2b512d8875a1fd1737dad3ef24a0a128a73a0/gym/envs/classic_control/pendulum.py
    new_θ = _angle_normalize(θ + new_dθ * p._dt)
    new_dθ = clamp(new_dθ, -p._max_speed, p._max_speed)

    p._state = Float32[new_θ, new_dθ]

    return _get_obs(p), reward(p), isterminal(p), γ(p)
end

function _get_obs(p::Pendulum)
    if p._trig_features
        θ, θ̇ = p._state
        return Float32[cos(θ), sin(θ), θ̇]
    else
        return p._state
    end
end

function reward(p::Pendulum)
    if p._sparse_rewards
        return  abs(p._state[1]) < π/12f0 ? 1f0 : 0f0
    else
        return cos(p._state[1])
    end
end

function isterminal(p::Pendulum)::Bool
    return false
end

function observation_space(p::Pendulum)
    p._observationspace
end

function action_space(p::Pendulum)
    p._actionspace
end

function γ(p::Pendulum)
    return p._γ
end

function _angle_normalize(θ)
    return (mod((θ + π), (2 * π)) - π)
end

function Base.show(io::IO, p::Pendulum)
    print(io, "Pendulum")
end

function _discrete_action(p::Pendulum, action)
    action = action[1]
    return p._max_torque * (action - 2)
end

function _continuous_action(p::Pendulum, action)
    return action[1]
end
