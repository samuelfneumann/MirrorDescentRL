using Random

mutable struct TwoArmBandit{
    A<:AbstractSpace,
    O<:AbstractSpace,
    R<:AbstractRNG,
} <: AbstractEnvironment
    const _obsspace::O
    const _actionspace::A

    _reward::Float32
    _state::Int
    _start_state::Int
    _done::Bool
    const _γ::Float32

    _Δ::Float32
    _stddev::Float32
    _rng::R

    function TwoArmBandit(rng::R, γ::Real, Δ::Real, σ::Real) where {R<:AbstractRNG}
        n_states = 30
        obsspace = Discrete(n_states)
        start_state = (n_states + mod(n_states, 2)) ÷ 2

        actionspace = Discrete(2)

        O = typeof(obsspace)
        A = typeof(actionspace)

        return new{A,O,R}(
            obsspace, actionspace, 0f0, start_state, start_state, false, γ, Δ, σ, rng,
        )
    end
end

function TwoArmBandit(rng::AbstractRNG; γ::Real, Δ::Real, σ::Real)
    return TwoArmBandit(rng, γ, Δ, σ)
end

function start!(b::TwoArmBandit)
    b._done = false
    b._state = b._start_state
    return [b._state]
end

function envstep!(b::TwoArmBandit, action)
    check_contains_action(b, action)

    b._state += (only(action) == 1 ? 1 : -1)
    b._reward = _reward(b, action)
    b._done = isterminal(b)

    return [b._state], reward(b), b._done, γ(b)
end

function reward(b::TwoArmBandit)
    return b._reward
end

function isterminal(b::TwoArmBandit)::Bool
    if b._state == 1
        @info "bad"
    elseif b._state == only(observation_space(b).n)
        @info "good"
    end
    return b._state == 1 || b._state == only(observation_space(b).n)
end

function γ(b::TwoArmBandit)
    return isterminal(b) ? 0f0 : b._γ
end

function observation_space(b::TwoArmBandit)
    return b._obsspace
end

function action_space(b::TwoArmBandit)::AbstractSpace
    return b._actionspace
end

function Base.show(io::IO, ::TwoArmBandit)
    print(io, "TwoArmBandit")
    return nothing
end

"""
    _reward(b::TwoArmBandit, action)

Computes the reward for taking action `action` in environment `b`.
"""
function _reward(b::TwoArmBandit, action)
    return if isterminal(b)
        r = (only(action) == 1 ? b._Δ : -b._Δ) + randn(b._rng) * b._stddev
    else
        -1f-3
    end
end
