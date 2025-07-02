@enum Integrator euler = 1 semi_implicit = 2

"""
    Cartpole <: AbstractEnvironment


The classic control Cartpole environment.

# Description
In this environment, a pole is attached to a cart. The cart starts with the pole nearly
upright, and the goal is to keep the pole in an upright position for as long as possible.

# State and Observations
The state observations consists of the `x` position of the cart, the cart's velocity, and
angle of the pole with respect to the cart, and the angular velocity of the pole. The
position of the cart is bounded between the `± x_threshold` constructor argument; the
velocity of the cart is bounded between the `± max_speed` argument; the angle of the pole is
ounded between `±theta_threshold_radians`;
and the angular velocity of the pole is between the `± max_angular_velocity` argument.

The episode is over when the pole angle falls below the `±theta_threshold_radians`
constructor argument.

# Actions
Actions may be discrete or continuous. If actions are discrete, then
3 actions are available:

Action | Meaning
-------|------------------------
   1   | Maximum acceleration of the cart leftwards
   2   | Nothing
   3   | Maximum acceleration of the cart rightward

If actions are continuous, then they are between
`[-max_action, max_action]`, where `-max_action` is maximum acceleration to the
left and `max_action` is maximum acceleration to the right; `max_accel` is given as a
constructor argument.

Given an action of `a ∈ [-max_action, max_action]` is taken, the effective action is `a *
force_mag`, where `force_mag` is the force magnifier, given as an argument to the
constructor.

# Goal/Rewards
The goal of the agent is to keep the pole within the angles `±theta_threshold_radians` as
long as possible, and the rewards are `1` on each timestep until the terminal timestep, at
which the reward is -1.

"""
mutable struct Cartpole{
        A<:AbstractSpace,
        O<:AbstractSpace,
        R<:AbstractRNG,
} <: AbstractEnvironment
    _gravity::Float32
    _cartmass::Float32
    _polemass::Float32
    _length::Float32 # Half the pole's length
    _force_mag::Float32
    _τ::Float32 # Seconds between state updates
    _integrator::Integrator

    # Threshold at which to end an episode
    _theta_threshold_radians::Float32 # Max/min angles of the pole
    _x_threshold::Float32 # Max/min x position of the cart
    _max_speed::Float32 # Max/min speed of the cart
    _max_angular_velocity::Float32 # Max/min angular velocity of the pole

    _obsspace::O
    _actionspace::A
    _continuous::Bool # Whether actions should be continuous or not

    _state::Vector{Float32}
    _γ::Float32
    _reward::Float32
    _rng::R
    _sparse_rewards::Bool

    function Cartpole(
        rng::R,
        continuous,
        γ,
        sparse_rewards,
    ) where {R}
        gravity = 9.8f0
        cartmass = 1f0
        polemass = 0.1f0
        length = 0.5f0
        force_mag = 10f0
        tau = 0.02f0
        integrator = euler
        theta_threshold_radians = 24f0 * π / 360f0
        x_threshold = 2.4f0
        max_speed = 2f0
        max_angular_velocity = 2f0
        max_action = 1f0

        high = [
            2f0 * x_threshold,
            max_speed,
            2f0 * theta_threshold_radians,
            max_angular_velocity,
        ]
        low = -high
        obs_space = Box{Float32}(low, high)
        action_space = continuous ? Box{Float32}(-max_action, max_action) : Discrete(3)

        O = typeof(obs_space)
        A = typeof(action_space)

        c = new{A,O,R}(
            gravity,
            cartmass,
            polemass,
            length,
            force_mag,
            tau,
            integrator,
            theta_threshold_radians,
            x_threshold,
            max_speed,
            max_angular_velocity,
            obs_space,
            action_space,
            continuous,
            [0f0, 0f0],
            γ,
            0f0,
            rng,
            sparse_rewards,
        )
        start!(c)
        return c
    end
end

function Cartpole(rng::AbstractRNG; continuous, γ = 1f0, sparse_rewards = false)
    return Cartpole(rng, continuous, γ, sparse_rewards)
end

function start!(c::Cartpole)
    # Random starting state in [-0.05, 0.05]^4
    c._state = rand(c._rng, Float32, 4) .* (0.1f0) .- 0.05f0
    return c._state
end

function envstep!(c::Cartpole, action)
    check_contains_action(c, action)

    # Convert action
    if !c._continuous
        a = _discrete_action(c, action)
    else
        a = _continuous_action(c, action)
    end

    # Magnify the force, which is already in [-1, 1]
    force = c._force_mag * a

    x, ẋ, θ, θ̇ = c._state

    cosθ = cos(θ)
    sinθ = sin(θ)

    temp = (force + _polemass_length(c) * θ̇^2 * sinθ) / _total_mass(c)
    θacc =
        (c._gravity * sinθ - cosθ * temp) /
        (c._length * (4f0 / 3f0 - c._polemass * cosθ^2f0 / _total_mass(c)))
    xacc = temp - _polemass_length(c) * θacc * cosθ / _total_mass(c)

    # Transition to the next state
    if c._integrator == euler
        x += (c._τ * ẋ)
        ẋ += (c._τ * xacc)
        θ += (c._τ * θ̇)
        θ̇ == (c._τ * θacc)
    elseif c._integrator == semi_implicit
        ẋ += (c._τ * xacc)
        x += (c._τ * ẋ)
        θ̇ == (c._τ * θacc)
        θ += (c._τ * θ̇)
    else
        error("no such integrator $c._integrator")
    end

    c._state = [x, ẋ, θ, θ̇]

    return c._state, reward(c), isterminal(c), γ(c)
end

function isterminal(c::Cartpole)::Bool
    x, _, θ, _ = c._state
    return (
        x < -c._x_threshold ||
        x > c._x_threshold ||
        θ > c._theta_threshold_radians ||
        θ < -c._theta_threshold_radians
    )
end

function γ(c::Cartpole)
    return isterminal(c) ? 0f0 : c._γ
end

function observation_space(c::Cartpole)
    return c._obsspace
end

function action_space(c::Cartpole)::AbstractSpace
    return c._actionspace
end

function reward(c::Cartpole)
    if c._sparse_rewards
        return isterminal(c) ? 1f0 : 0f0
    else
        return isterminal(c) ? 1f0 : -1f0
    end
end

function Base.show(io::IO, c::Cartpole)
    type = continuous(action_space(c)) ? "Continuous" : "Discrete"
    print(io, "Cartpole")
end

"""
    _polemass_length(c::Cartpole)::Float32

Return the linear density of the pole
"""
function _polemass_length(c::Cartpole)
    return c._polemass * c._length
end

"""
    _total_mass(c::Cartpole)::Float32

Return the total mass of the cartpole
"""
function _total_mass(c::Cartpole)
    return c._polemass + c._cartmass
end

function _discrete_action(::Cartpole, a)::Int
    return a[1] - 2 # Convert to be in (-1, 0, 1)
end

function _continuous_action(c::Cartpole, a)
    lower = c |> action_space |> low
    upper = c |> action_space |> high
    return a[1]
end
