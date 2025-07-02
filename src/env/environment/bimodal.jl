"""
    Bimodal <: AbstractEnvironment

Implements a bandit problem with a bimodal reward function.

# Description
Bimodal implements a bandit problem with a bimodal reward function. The reward function
itself is a pdf of a mixture of Gaussians with a lower mode at -1.0 and a higher mode at
1.0. The modes have y values of 1.0 and 1.5 respectively. Both Gaussians have the
same standard deviation. Given some action, the reward consists of the pdf of this action
under this distribution.

This environment is completely deterministic.
# State and Observations
Since this is a bandit problem, there is no concept of state. Even so, function
approximators need some kind of input, therefore the only "state observation" is `[1]`, a
vector with a single element, which is `1`.

# Actions
Actions are bounded between `[-2, 2]`.

# Goal/Rewards
Let `N(a; μ, σ)` be the normal density function evaluated at `a` for a normal distribution
with mean `μ` and standard deviation `σ`. Then, the reward function `r(a)` is given by:

    r(a) = 1.0 * N(a; -1.0, 0.2) + 1.5 * N(a; 1.0, 0.2)
"""
mutable struct Bimodal{
        A<:AbstractSpace,
        O<:AbstractSpace,
} <: AbstractEnvironment
    _obsspace::O
    _actionspace::A

    _reward::Float32
    _state::Vector{Float32}
    _done::Bool
    _γ::Float32

    # y values for modes of reward function
    _mode1::Float32
    _mode2::Float32

    # Standard deviation of reward function
    _stddev::Float32

    function Bimodal()
        high_obs = 1f0
        low_obs = high_obs
        obsspace = Box{Float32}([low_obs], [high_obs])
        state = [high_obs]

        high_action = 2f0
        low_action = -high_action
        actionspace = Box{Float32}([low_action], [high_action])

        γ = 0f0 # Bandit problem

        O = typeof(obsspace)
        A = typeof(actionspace)

        return new{A,O}(obsspace, actionspace, 0f0, state, false, γ, 1f0, 1.5f0, 0.2f0)
    end
end

function start!(b::Bimodal)
    b._done = false
    return b._state
end

function envstep!(b::Bimodal, action)
    check_contains_action(b, action)

    b._reward = _reward(b, action)
    b._done = true

    return b._state, reward(b), true, γ(b)
end

function reward(b::Bimodal)
    return b._reward
end

function isterminal(b::Bimodal)::Bool
    return b._done
end

function γ(b::Bimodal)
    return b._γ
end

function observation_space(b::Bimodal)
    return b._obsspace
end

function action_space(b::Bimodal)::AbstractSpace
    return b._actionspace
end

function Base.show(io::IO, ::Bimodal)
    print(io, "Bimodal")
    return nothing
end

"""
    _reward(b::Bimodal, action)

Computes the reward for taking action `action` in environment `b`.
"""
function _reward(b::Bimodal, action)
    action_range = high(action_space(b)) .- low(action_space(b))
    step = action_range / 4f0

    maxima1 = low(action_space(b)) .+ step
    maxima2 = maxima1 .+ 2f0 .* step

    modal1 = b._mode1 * exp.(-0.5f0 * ((action .- maxima1) / b._stddev).^2f0)
    modal2 = b._mode2 * exp.(-0.5f0 * ((action .- maxima2) / b._stddev).^2f0)

    reward =  modal1 .+ modal2

    return reward[1]
end
