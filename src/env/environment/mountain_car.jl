"""
    MountainCar <: AbstractEnvironment

# Description
The Mountain Car MDP is a deterministic MDP that consists
of a car placed stochastically at the bottom of a sinusoidal valley, with
the only possible actions being the accelerations that can be applied to
the car in either direction. The goal of the MDP is to strategically
accelerate the car to reach the goal state on top of the right hill.

This MDP first appeared in [Andrew Moore's PhD Thesis
(1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
```
@PhDThesis{Moore90efficientmemory-based,
    author = {Andrew William Moore},
    title = {Efficient Memory-based Learning for Robot Control},
    institution = {University of Cambridge},
    year = {1990}
}
```

## Observation Space
The observation is a `ndarray` with shape `(2,)` where the elements
correspond to the following:
| Num | Observation            | Min   | Max  |
|-----|------------------------|-------|------|
| 0   | x-position of the car  | -1.2  | 0.6  |
| 1   | velocity of the car    | -0.07 | 0.07 |

## Action Space
In the continuous-action setting, the action is a `ndarray` with shape
`(1,)`, representing the directional force applied on the car.
The action is clipped in the range `[-1,1]` and multiplied by the force
(see the section on transition dynamics. In the discrete action setting,
the action is an integer in the set `{0, 1, 2}` corresponding to maximum
negative acceleration, no acceleration, and maximum positive acceleration.

## Transition Dynamics:
Given an action, the mountain car follows the following transition
dynamics: *velocity<sub>t+1</sub> = velocity<sub>t+1</sub> + force *
self.power - 0.0025 * cos(3 * position<sub>t</sub>)*
*position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
where force is the action clipped to the range `[-1,1]` and power is a
constant:
    - 0.001 in the discrete-action setting
    - 0.0015 in the continuous-action setting if `gym_version == True` else
        0.001
The collisions at either end are inelastic with the
velocity set to 0 upon collision with the wall.  The position is clipped to
the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

## Reward
The reward is -1 per step if `sparse_rewards == False`. Otherwise it is 0
per step, with a +1 given for reaching the goal state.

## Starting State
The position of the car is assigned a uniform random value in `[-0.6 ,
-0.4]`.  The starting velocity of the car is always assigned to 0.

## Episode End
The episode ends if the position of the car is greater than or equal to the
goal position on the right-hand-side hill which is:
    - 0.5 in the discrete-action setting
    - 0.45 in the continuous-action setting if `gym_version == True` else
        0.5
"""
mutable struct MountainCar{
        A<:AbstractSpace,
        O<:AbstractSpace,
        R<:AbstractRNG,
} <: AbstractEnvironment
    const _observationspace::O
    const _actionspace::A
    const _γ::Float32

    const _sparse_rewards::Bool

    const _minpos::Float32
    const _maxpos::Float32
    const _maxspeed::Float32
    const _goalpos::Float32

    const _power::Float32
    const _gravity::Float32

    _rng::R
    _state::Vector{Float32}

    function MountainCar(
        rng::R,
        γ,
        continuous::Bool,
        sparse_rewards,
        gym_version,
    ) where {R<:AbstractRNG}
        min_pos = -1.2f0
        max_pos = 0.6f0
        max_speed = 0.07f0
        min_speed = -max_speed
        gravity = 0.0025

        if continuous
            max_accel = 1.0f0
            min_action = -max_accel
            max_action = max_accel
            action_space = Box{Float32}(min_action, max_action)
            goal_pos = gym_version ? 0.45 : 0.5
            power = gym_version ? 0.0015 : 0.001
        else
            goal_pos = 0.5
            power = 0.001
            action_space = Discrete{Int64}(3)
        end

        obs_space = Box{Float32}([min_pos, min_speed], [max_pos, max_speed])
        O = typeof(obs_space)
        A = typeof(action_space)

        m = new{A,O,R}(
            obs_space,
            action_space,
            γ,
            sparse_rewards,
            min_pos,
            max_pos,
            max_speed,
            goal_pos,
            power,
            gravity,
            rng,
            Float32[0f0, 0f0],
        )

        start!(m)
        return m
    end
end

function MountainCar(
    rng::AbstractRNG;
    γ = 1.0f0,
    continuous::Bool,
    sparse_rewards = false,
    gym_version = true,
)
    return MountainCar(rng, γ, continuous, sparse_rewards, gym_version)
end

function start!(m::MountainCar)
    high = -0.4f0
    low = -0.6f0
    position = rand(m._rng, Float32) * (high - low) + low

    m._state = Float32[position, 0f0]
    return m._state
end

function envstep!(m::MountainCar, action)
    check_contains_action(m, action)

    # Sanitize the input action depending on whether we are in the
    # discrete or continuous setting
    if !(m |> action_space |> continuous)
        force = _discrete_action(m, action)
    else
        force = _continuous_action(m, action)
    end

    # Update the velocity
    position, velocity = m._state
    velocity += force * m._power + cos(3f0 * position) * (-m._gravity)
    # TODO: should we instead get these bounds from the observationspace?
    velocity = clamp(velocity, -m._maxspeed, m._maxspeed)

    # Update the position
    position += velocity
    position = clamp(position, m._minpos, m._maxpos)
    if position == m._minpos && velocity < 0.0f0
        velocity = 0.0f0
    end
    m._state = Float32[position, velocity]

    return m._state, reward(m), isterminal(m), γ(m)
end

function reward(m::MountainCar)
    if m._sparse_rewards
        reward = isterminal(m) ? 1.0f0 : 0.0f0
    else
        reward = -1.0f0
    end

    return reward
end

function isterminal(m::MountainCar)::Bool
    return m._state[1] >= m._goalpos
end

function observation_space(m::MountainCar)
    m._observationspace
end

function action_space(m::MountainCar)
    m._actionspace
end

function γ(m::MountainCar)
    isterminal(m) ? 0.0f0 : m._γ
end

function Base.show(io::IO, m::MountainCar)
    print(io, "MountainCar")
end

function _discrete_action(m::MountainCar, action)::Int
    action = action[1]
    action -= 2
end

function _continuous_action(m::MountainCar, action)
    action[1]
end
