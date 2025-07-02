mutable struct NoisyCliffWorld{
        T<:Real,
        A<:AbstractSpace,
        O<:AbstractSpace,
        R<:AbstractRNG,
} <: AbstractEnvironment
    const _observationspace::O
    const _actionspace::A
    const _γ::Float32
    _rng::R
    _current_state::Int
    const _rows::Int
    const _cols::Int
    const _int_obs::Bool
    const _exploring_starts::Bool
    _last_transition_off_cliff::Bool

    function NoisyCliffWorld{T}(
        rng::R, γ, rows, cols, int_obs, exploring_starts,
    ) where {T<:Real,R<:AbstractRNG}

        if int_obs
            obs_space = Discrete{Int}(rows * cols)
        else
            low = zeros(T, rows * cols)
            high = ones(T, rows * cols)
            obs_space = Box{Int}(low, high)
        end

        action_space = Discrete(4)
        O = typeof(obs_space)
        A = typeof(action_space)

        p = new{T,A,O,R}(
            obs_space, action_space, γ, rng, rows, rows, cols, int_obs, exploring_starts,
            false,
        )

        start!(p)
        return p
    end
end

function NoisyCliffWorld(
        rng::AbstractRNG; γ=1f0, rows=4, cols=12, int_obs=true, exploring_starts=false,
)
    NoisyCliffWorld{Int}(
        rng; γ=γ, rows=rows, cols=cols, int_obs=int_obs, exploring_starts=exploring_starts,
    )
end

function NoisyCliffWorld{T}(
    rng::AbstractRNG; γ=1f0, rows=4, cols=12, int_obs=true, exploring_starts=false,
) where {T<:Real}
    NoisyCliffWorld{T}(rng, γ, rows, cols, int_obs, exploring_starts)
end

function _to_grid(c::NoisyCliffWorld{T}; vec=false) where {T}
    if vec
        grid = spzeros(T, c._rows * c._cols)
    else
        grid = spzeros(T, c._rows, c._cols)
    end
    grid[c._current_state] = one(T)
    return grid
end

function reward(c::NoisyCliffWorld)
    return if c._last_transition_off_cliff
        -100f0
    elseif _at_goal(c)
        ε = rand(Float32)
        q = -1000f0
        thresh = (abs(q) - 88 + 1) / (abs(q) - 1)
        r = ε > thresh ? q : -1f0
        if ε > thresh
            @show r
            sleep(1)
        end
        r
    else
        -1f0
    end
end

γ(c::NoisyCliffWorld) = isterminal(c) ? zero(c._γ) : c._γ
observation_space(c::NoisyCliffWorld) = c._observationspace
action_space(c::NoisyCliffWorld) = c._actionspace

function isterminal(c::NoisyCliffWorld)
    return _at_goal(c) || c._last_transition_off_cliff
end

function start!(c::NoisyCliffWorld{T}) where {T}
    if c._exploring_starts
        non_cliff_states = []
        for i in 1:observation_space(c).n[1]
            if !_on_cliff(c, i)
                push!(non_cliff_states, i)
            end
        end
        c._current_state = rand(c._rng, non_cliff_states)
    else
        c._current_state = c._rows
    end
    c._last_transition_off_cliff = false

    return _get_obs(c)
end

function _to_index(c::NoisyCliffWorld)
    return _to_index(c, c._current_state)
end

function _to_index(c::NoisyCliffWorld, i)
    row = ((i - 1) - c._rows * ((i - 1) ÷ c._rows)) + 1
    col = ((i - 1) ÷ c._rows) + 1

    return (col, row)
end

_up(c::NoisyCliffWorld) = -1
_down(c::NoisyCliffWorld) = 1
_right(c::NoisyCliffWorld) = c._rows
_left(c::NoisyCliffWorld) = -c._rows

function envstep!(c::NoisyCliffWorld, action)
    check_contains_action(c, action)

    u = _discrete_action(c, action)

    last_state = c._current_state
    if _in_first_col(c) && u == _left(c)
    elseif _in_last_col(c) && u == _right(c)
    elseif _in_first_row(c) && u == _up(c)
    elseif _in_last_row(c) && u == _down(c)
    else
        c._current_state += u
    end

    # Cache whether the agent jumped off the cliff
    c._last_transition_off_cliff = _on_cliff(c)

    # If the agent did jump off the cliff, transition to the start state
    if _on_cliff(c)
        c._current_state = c._rows
    end

    return _get_obs(c), reward(c), isterminal(c), γ(c)
end

function Base.show(io::IO, c::NoisyCliffWorld{T}) where {T}
    print(io, "NoisyCliffWorld{$T}")
end

function _get_obs(c::NoisyCliffWorld{T}) where {T}
    return if c._int_obs
        return [c._current_state]
    else
        _to_grid(c)
    end
end

_at_goal(c::NoisyCliffWorld) = c._current_state == c._cols * c._rows
_at_goal(c::NoisyCliffWorld, col::Int, row::Int) = col * row == c._cols * c._rows

function _on_cliff(c::NoisyCliffWorld)
    col, row = _to_index(c)
    return _on_cliff(c, col, row)
end

function _on_cliff(c::NoisyCliffWorld, col::Int, row::Int)
    return 1 < col < c._cols && row == c._rows
end

function _on_cliff(c::NoisyCliffWorld, i::Int)
    _on_cliff(c, _to_index(c, i)...)
end

function _discrete_action(c::NoisyCliffWorld, action)
    if action isa AbstractArray
        @assert length(action) == 1
        action = first(action)
    end
    actions = [_up, _down, _right, _left]
    return actions[action](c)
end

function _in_first_col(c::NoisyCliffWorld)
    return c._current_state <= c._rows
end

function _in_last_col(c::NoisyCliffWorld)
    return (
        c._current_state > (c._rows * c._cols) - c._rows
    )
end

function _in_first_row(c::NoisyCliffWorld)
    return mod(c._current_state, c._rows) == 1
end

function _in_last_row(c::NoisyCliffWorld)
    return mod(c._current_state, c._rows) == 0
end
