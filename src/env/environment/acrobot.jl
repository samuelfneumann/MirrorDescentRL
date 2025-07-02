"""
    Acrobot <: AbstractEnvironment

## Description
The Acrobot environment is based on Sutton's work in [1] and [2].
The system consists of two links connected linearly to form a chain,
with one end of the chain fixed. The joint between the two links is
actuated. The goal is to apply torques on the actuated joint to swing the
free end of the linear chain above a given height while starting from the
initial state of hanging straight downwards with no velocity.

## Observation Space
The observation is a `ndarray` with shape `(4,)`:
| Num | Observation | Min     | Max    |
|-----|-------------|---------|--------|
| 0   | θ₁          | -π      | π      |
| 1   | θ₂          | -π      | π      |
| 2   | θ̇₁          | -4 * pi | 4 * pi |
| 3   | θ̇₂          | -9 * pi | 9 * pi |
where θ₁ is the angle of the first joint, where an angle of 0 indicates the
first link is pointing directly downwards. θ₂ is the angle of the second
joint, **relative** to the angle of the first link. An angle of 0
corresponds to having the same angle between the two links.
The angular velocities, θ̇₁ and θ̇₂, are bounded at ±4π, and ±9π rad/s
respectively.

## Action Space
In the continuous-action setting, the action is a `ndarray` with shape
`(1,)` representing the torque applied on the actuated joint between the
two links
| Name | Action | Min  | Max |
|------|--------|------|-----|
| 0    | Torque | -1.0 | 1.0 |

In the discrete-action setting, the action is an integer in the set
`{0, 1, 2}`, corresponding to maximum torque in the negative direction,
no torque, maximum torque in the positive direction.

## Rewards
The reward is -1 per step if `sparse_rewards == False`. Otherwise it is 0
per step, with a +1 given for reaching the goal state.

## Starting State
The starting state is facing straight down with 0 angular velocity: [0, 0,
0, 0].

## Episode End
The episode ends if the free end reaches the target height, which is
constructed as: `-cos(theta1) - cos(theta2 + theta1) > 1.0`

## Transition Dynamics
By default, the dynamics of the acrobot follow those described in Sutton
and Barto's book [2].

## Additional Notes
This version of the domain uses the Runge-Kutta method for integrating
the system dynamics and is more realistic, but also considerably harder
than the original version which employs Euler integration.

## References
[1] Sutton, R. S. (1996). Generalization in Reinforcement Learning:
    Successful Examples Using Sparse Coarse Coding.  In D. Touretzky, M. C.
    Mozer, & M. Hasselmo (Eds.), Advances in Neural Information Processing
    Systems (Vol. 8). MIT Press.
[2] Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An
Introduction. The MIT Press.
"""
mutable struct Acrobot{
    A<:AbstractSpace,
    O<:AbstractSpace,
    R<:AbstractRNG,
} <: AbstractEnvironment
    const _observationspace::O
    const _actionspace::A
    const _γ::Float32
    const _continuous::Bool

    const _sparse_rewards::Bool

    const _link_length_1::Float32
    const _link_length_2::Float32
    const _link_mass_1::Float32  # mass in kg
    const _link_mass_2::Float32  # mass in kg
    const _link_com_pos_1::Float32  # Centre of mass
    const _link_com_pos_2::Float32  # Centre of mass
    const _link_moi::Float32   # Moments of inertia
    const _threshold::Float32

    const _max_vel_1::Float32
    const _max_vel_2::Float32
    const _action_noise::Float32
    const _gravity::Float32

    _rng::R

    _state::Vector{Float32}
    const _dt::Float32
    const _book_over_nips::Bool

    function Acrobot(
        rng::R,
        continuous,
        sparse_rewards,
        γ,
    ) where {R<:AbstractRNG}

        link_length_1 = 1f0
        link_mass_1 = 1f0
        link_com_pos_1 = 0.5f0
        link_length_2 = 1f0
        link_mass_2 = 1f0
        link_com_pos_2 = 0.5f0
        link_moi = 1f0
        threshold = 1.0f0
        # Whether to use the implementation described in the book or the NeurIPS conference paper
        book_over_nips = true
        max_vel_1 = 4f0 * π
        max_vel_2 = 9f0 * π
        max_torque = 1f0
        action_noise = 0f0
        dt = 0.2f0
        gravity = 9.8f0

        high = Float32[π, π, max_vel_1, max_vel_2]
        obs_space = Box{Float32}(-high, high)
        action_space = continuous ? Box{Float32}(-max_torque, max_torque) : Discrete(3)
        O = typeof(obs_space)
        A = typeof(action_space)

        a = new{A,O,R}(
            obs_space,
            action_space,
            γ,
            continuous,
            sparse_rewards,
            link_length_1,
            link_length_2,
            link_mass_1,
            link_mass_2,
            link_com_pos_1,
            link_com_pos_2,
            link_moi,
            threshold,
            max_vel_1,
            max_vel_2,
            action_noise,
            gravity,
            rng,
            zeros(Float32, size(obs_space)[1]),
            dt,
            book_over_nips,
        )


        start!(a)
        return a
    end
end

function Acrobot(rng::AbstractRNG; continuous, γ=1.0f0, sparse_rewards=false)
    return Acrobot(rng, continuous, sparse_rewards, γ)
end


function start!(a::Acrobot)
    a._state = zero(observation_space(a) |> low)
    return a._state
end

function envstep!(a::Acrobot, action)
    check_contains_action(a, action)

    # Sanitize the input action depending on whether we are in the
    # discrete or continuous setting
    if !(a |> action_space |> continuous)
        torque = action[1]
        torque -= 2  # ∈ [-1, 0, 1]
    else
        torque = action[1]
    end

    if a._action_noise > 0f0
        noise = rand(a._rng, P)
        noise_min = -a._action_noise
        noise_max = a._action_noise
        noise = noise_min + noise * (noise_max - noise_min)
        torque += noise
    end

    # Augment the state with the torque so it can be passed to _dsdt
    s = a._state
    s_augmented = [s..., torque]

    ns = _rk4(a, _dsdt, s_augmented, Float32[0f0, a._dt])
    ns[1] = wrap(ns[1], -π, π)
    ns[2] = wrap(ns[2], -π, π)
    ns[3] = bound(ns[3], -a._max_vel_1, a._max_vel_1)
    ns[4] = bound(ns[4], -a._max_vel_2, a._max_vel_2)

    a._state = ns

    return a._state, reward(a), isterminal(a), γ(a)
end

function reward(a::Acrobot)
    if a._sparse_rewards
        return isterminal(a) ? 1f0 : 0f0
    else
        return -1f0
    end
end

function isterminal(a::Acrobot)::Bool
    s = a._state
    return -cos(s[1]) - cos(s[2] + s[1]) > a._threshold
end

function observation_space(a::Acrobot)
    a._observationspace
end

function action_space(a::Acrobot)
    a._actionspace
end

function γ(a::Acrobot)
    isterminal(a) ? 0f0 : a._γ
end

function Base.show(io::IO, a::Acrobot)
    print(io, "Acrobot")
end

"""
	_dsdt(acro::Acrobot, s_augmented)

Calculate the derivatives of the physical constants in the environment `acro`.
"""
function _dsdt(acro::Acrobot, s_augmented)
    m1 = acro._link_mass_1
    l1 = acro._link_length_1
    lc1 = acro._link_com_pos_1
    m2 = acro._link_mass_2
    l2 = acro._link_length_2
    lc2 = acro._link_com_pos_2
    I1 = acro._link_moi
    I2 = acro._link_moi
    g = acro._gravity
    a = s_augmented[end]
    s = s_augmented[begin:end-1]
    θ1 = s[1]
    θ2 = s[2]
    dθ1 = s[3]
    dθ2 = s[4]

    d1 = m1 * lc1^2
    d1 += m2 * (l1^2 + lc2^2 + 2 * l1 * lc2 * cos(θ2))
    d1 += I1
    d1 += I2

    d2 = m2 * (lc2^2 + l1 * lc2 * cos(θ2)) + I2

    ϕ2 = m2 * lc2 * g * cos(θ1 + θ2 - π/2f0)
    ϕ1 = -m2 * l1 * lc2 * dθ2^2 * sin(θ2)
    ϕ1 += -2m2 * l1 * lc2 * dθ2 * dθ1 * sin(θ2)
    ϕ1 += (m1 * lc1 + m2 * l1) * g * cos(θ1 - π/2f0)
    ϕ1 += ϕ2

    if ! acro._book_over_nips
        # Nips
        ddθ2 = a + d2 / d1 * ϕ1 - ϕ2
        ddθ2 /= (m2 * lc2^2 + I2 - d2^2 / d1)
    else
        # Book
        ddθ2 = a + d2 / d1 * ϕ1 - m2 * l1 * lc2 * dθ1^2 * sin(θ2) - ϕ2
        ddθ2 /= (m2 * lc2^2 + I2 - d2^2 / d1)
    end

    ddθ1 = -(d2 * ddθ2 + ϕ1) / d1

    return dθ1, dθ2, ddθ1, ddθ2, 0.0
end

"""
	_rk4(a::Acrobot, derivs, y0, t, P::Type{<:AbstractFloat})

Integrate the 1D or ND system of ODEs using 4th order Runge-Kitta.

# Arguments
- `a::Acrobot`: the acrobot environment to get the physical constants from
- `derivs`: a function which computes the derivatives of the system, which has the signature
`dy = derivs(yi)`
- `t`: a vector of sample times
- `P::Type{<:AbstractFloat}`: the floating precision to use
"""
function _rk4(a::Acrobot, derivs, y0, t)
    Ny = length(y0)
    yout = zeros(Float32, length(t), Ny)

    yout[1, :] .= y0

    for i = 1:length(t)-1
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2f0
        y0 = yout[i, :]

        k1 = [derivs(a, y0)...]
        k2 = [derivs(a, y0 .+ dt2 .* k1)...]
        k3 = [derivs(a, y0 .+ dt2 .* k2)...]
        k4 = [derivs(a, y0 .+ dt .* k3)...]
        yout[i + 1, :] .= y0 .+ dt/6f0 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end

    # We only care about the final timestep, and cleave off action value which will be 0
    return yout[end, 1:4]

end

"""
	bound(x, m[, M=nothing])

Clamp `x` to be between `m` and `M`. If `M` is `nothing`, clamp `x` to be between `m[1]` and
m[2]`.
"""
function bound(x, m, M=nothing)
    if M === nothing
        M = m[2]
        m = m[1]
    end

    return clamp.(x, m, M)
end

"""
	wrap(x, m, M)

Wrap `x` so that `x ∈ [m, M]` where `x`, `m`, and `M` are all scalars.
"""
function wrap(x, m, M)
    diff = M - m

    while x > M
        x -= diff
    end

    while x < m
        x += diff
    end

    return x
end
