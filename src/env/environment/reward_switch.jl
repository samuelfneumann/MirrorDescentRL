# For a tabular policy, states will be inputted as Vector{Bool}, Vector{Int} or {Int},
# actions are Int
const REWARD_SWITCH_N_STATES = 7

mutable struct RewardSwitch{
        T<:Real,
        A<:AbstractSpace,
        O<:AbstractSpace,
        F,
        R<:AbstractRNG,
} <: AbstractEnvironment
    _rng::R
    _observationspace::O
    _actionspace::A
    _γ::Float32
    _current_step::Int
    _switch_at::Int
    _current_state::Int
    _last_action::Int
    _int_obs::Bool
    _second_stage::Bool
    _action_trans::F

    function RewardSwitch{T}(rng::R, γ, switch_at, int_obs) where {T<:Real,R<:AbstractRNG}
        @assert switch_at > 0

        if int_obs
            obs_space = Discrete{Int}(REWARD_SWITCH_N_STATES)
        else
            low = zeros(T, REWARD_SWITCH_N_STATES)
            high = ones(T, REWARD_SWITCH_N_STATES)
            obs_space = Box{Int}(low, high)
        end

        action_space = Discrete(2)
        O = typeof(obs_space)
        A = typeof(action_space)

        top_action_trans = Dict(
            1 => 2,
            2 => 4,
            4 => 1,
            5 => 1,
            3 => 6,
            6 => 1,
            7 => 1,
        )

        bottom_action_trans = Dict(
            1 => 3,
            3 => 7,
            7 => 1,
            6 => 1,
            2 => 5,
            7 => 1,
            4 => 1,
        )

        action_trans = (top_action_trans, bottom_action_trans)
        F = typeof(action_trans)

        p = new{T,A,O,F,R}(
            rng, obs_space, action_space, γ, 1, switch_at, 1, -1, int_obs, false,
            action_trans,
        )

        start!(p)
        return p
    end
end

function RewardSwitch(rng::AbstractRNG; γ=1f0, switch_at, int_obs=true)
    RewardSwitch{Int}(rng; γ=γ, switch_at=switch_at, int_obs=int_obs)
end

function RewardSwitch{T}(rng; γ=1f0, switch_at, int_obs=true) where {T<:Real}
    RewardSwitch{T}(rng, γ, switch_at, int_obs)
end

function reward(r::RewardSwitch)
    if r._last_action < 1
        @warn "should not call reward unless action has been taken" maxlog=1
        return 0f0
    end

    p = rand(r._rng)

    top_r1 = -1f0
    top_r2 = -1000f0
    bottom_r = -10f0

    return if r._second_stage
        if r._last_action == 1
            if p < (1f0 - (bottom_r - top_r2) / (top_r1 - top_r2)) / 1.5f0
                top_r2
            else
                top_r1
            end
        else
            bottom_r
        end
    else
        r._last_action == 1 ? top_r1 : bottom_r
    end
end

γ(r::RewardSwitch) = isterminal(r) ? zero(r._γ) : r._γ
observation_space(r::RewardSwitch) = r._observationspace
action_space(r::RewardSwitch) = r._actionspace
isterminal(r::RewardSwitch) = _at_goal(r)
_at_goal(r::RewardSwitch) = r._current_state in (4, 5, 6, 7)

function start!(r::RewardSwitch{T}) where {T}
    r._current_state = 1
    r._last_action = -1

    return _get_obs(r)
end

function envstep!(r::RewardSwitch, action)
    check_contains_action(r, action)
    @assert length(action) == 1 # actions are integers
    action = action[1]
    r._current_step += 1
    r._current_state = r._action_trans[action][r._current_state]
    r._last_action = action

    if r._current_step == r._switch_at
        r._second_stage = true
    end

    return _get_obs(r), reward(r), isterminal(r), γ(r)
end

function _get_obs(r::RewardSwitch{T}) where {T}
    return if r._int_obs
        [r._current_state]
    else
        z = spzeros(T, REWARD_SWITCH_N_STATES)
        z[r._current_state] = one(T)
    end
end

function Base.show(io::IO, r::RewardSwitch{T}) where {T}
    s1 = """
            O
        O - |
       |    O
    x -
       |    O
        O - |
            O
    """

    s2 = """
            O
        x - |
       |    O
    O -
       |    O
        O - |
            O
    """

    s3 = """
            O
        O - |
       |    O
    O -
       |    O
        x - |
            O
    """

    s4 = """
            x
        O - |
       |    O
    O -
       |    O
        O - |
            O
    """

    s5 = """
            O
        O - |
       |    x
    O -
       |    O
        O - |
            O
    """

    s6 = """
            O
        O - |
       |    O
    O -
       |    x
        O - |
            O
    """

    s7 = """
            O
        O - |
       |    O
    O -
       |    O
        O - |
            x
    """

    print(io, [s1, s2, s3, s4, s5, s6, s7][r._current_state])
    println(io, "RewardSwitch{$T}")
end

