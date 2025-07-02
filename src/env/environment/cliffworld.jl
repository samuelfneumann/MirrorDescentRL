mutable struct CliffWorld{
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

    function CliffWorld{T}(
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

function CliffWorld(
    rng::AbstractRNG; γ=1f0, rows=4, cols=12, int_obs=true, exploring_starts=false,
)
    CliffWorld{Int}(
        rng; γ=γ, rows=rows, cols=cols, int_obs=int_obs, exploring_starts=exploring_starts,
    )
end

function CliffWorld{T}(
    rng::AbstractRNG; γ=1f0, rows=4, cols=12, int_obs=true, exploring_starts=false,
) where {T<:Real}
    CliffWorld{T}(rng, γ, rows, cols, int_obs, exploring_starts)
end

function _to_grid(c::CliffWorld{T}, i::Int; vec=false) where {T}
    if vec
        grid = spzeros(T, c._rows * c._cols)
    else
        grid = spzeros(T, c._rows, c._cols)
    end
    grid[i] = one(T)
    return grid
end

function _to_grid(c::CliffWorld{T}; vec=false) where {T}
    return _to_grid(c, c._current_state)
end

function _to_grid(c::CliffWorld{T}, col::Int, row::Int; vec=false) where {T}
    return _to_grid(c, _rowcol2index(c, col, row))
end

function reward(c::CliffWorld, s_t_index::Int, a_t)
    s_tp1_index = __next_state(c, s_t_index, a_t)
    return _on_cliff(c, s_tp1_index) ? -100f0 : -1f0
end

reward(c::CliffWorld) = c._last_transition_off_cliff ? -100f0 : -1f0
γ(c::CliffWorld) = isterminal(c) ? zero(c._γ) : c._γ
observation_space(c::CliffWorld) = c._observationspace
action_space(c::CliffWorld) = c._actionspace
isterminal(c::CliffWorld) = _at_goal(c)

function start!(c::CliffWorld{T}) where {T}
    if c._exploring_starts
        non_cliff_states = []
        for i in 1:observation_space(c).n[1]
            if !_on_cliff(c, i) && !_at_goal(c, i)
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

function _index2rowcol(c::CliffWorld)
    return _index2rowcol(c, c._current_state)
end

function _index2rowcol(c::CliffWorld, i)
    @assert 1 <= i <= c._rows * c._cols
    row = ((i - 1) - c._rows * ((i - 1) ÷ c._rows)) + 1
    col = ((i - 1) ÷ c._rows) + 1

    return (col, row)
end

function _rowcol2index(c::CliffWorld, col::Int, row::Int)
    @assert 1 <= row <= c._rows
    @assert 1 <= col <= c._cols
    return (col - 1) * c._rows + row
end

_up(c::CliffWorld) = -1
_down(c::CliffWorld) = 1
_right(c::CliffWorld) = c._rows
_left(c::CliffWorld) = -c._rows

function _next_state(c::CliffWorld, s_t_index::Int, a_t)
    next_state = __next_state(c, s_t_index, a_t)
    return _on_cliff(c, next_state) ? c._rows : next_state
end

function __next_state(c::CliffWorld, s_t_index::Int, a_t)
    u = _discrete_action(c, a_t)

    return if _on_cliff(c, s_t_index)
        return c._rows
    elseif _in_first_col(c, s_t_index) && u == _left(c)
        s_t_index
    elseif _in_last_col(c, s_t_index) && u == _right(c)
        s_t_index
    elseif _in_first_row(c, s_t_index) && u == _up(c)
        s_t_index
    elseif _in_last_row(c, s_t_index) && u == _down(c)
        s_t_index
    else
        s_t_index += u
    end
end

function _next_state(c::CliffWorld, s_t_col::Int, s_t_row::Int, a_t)
    return _index2rowcol(c, _next_state(c, _rowcol2index(c, s_t_col, s_t_row)...))
end

function _next_state_and_reward(c::CliffWorld, s_t_index::Int, a_t)
    next_state = _next_state(c, s_t_index, a_t)
    r = reward(c, s_t_index, a_t)
    return next_state, r
end

function envstep!(c::CliffWorld, a_t)
    check_contains_action(c, a_t)

    last_state = c._current_state
    c._current_state = _next_state(c, c._current_state, a_t)

    # Cache whether the agent jumped off the cliff
    c._last_transition_off_cliff = _on_cliff(c)

    # If the agent did jump off the cliff, transition to the start state
    if _on_cliff(c)
        c._current_state = c._rows
    end

    return _get_obs(c), reward(c), isterminal(c), γ(c)
end

function Base.show(io::IO, c::CliffWorld{T}) where {T}
    print(io, "CliffWorld{$T}")
end

function _get_obs(c::CliffWorld{T}, i::Int) where {T}
    return if c._int_obs
        @assert 1 <= i <= c._cols * c._rows
        return [i]
    else
        _to_grid(c, i)
    end
end

function _get_obs(c::CliffWorld{T}) where {T}
    return _get_obs(c, c._current_state)
end

function _get_obs(c::CliffWorld{T}, col::Int, row::Int) where {T}
    return _get_obs(c, _rowcol2index(c, col, row))
end

_at_goal(c::CliffWorld) = c._current_state == c._cols * c._rows
_at_goal(c::CliffWorld, col::Int, row::Int) = col * row == c._cols * c._rows
_at_goal(c::CliffWorld, i::Int) = _at_goal(c, _index2rowcol(c, i)...)

function _on_cliff(c::CliffWorld)
    col, row = _index2rowcol(c)
    return _on_cliff(c, col, row)
end

function _on_cliff(c::CliffWorld, col::Int, row::Int)
    return 1 < col < c._cols && row == c._rows
end

function _on_cliff(c::CliffWorld, i::Int)
    _on_cliff(c, _index2rowcol(c, i)...)
end

function _discrete_action(c::CliffWorld, a_t)
    if a_t isa AbstractArray
        @assert length(a_t) == 1
        a_t = first(a_t)
    end
    actions = [_up, _down, _right, _left]
    return actions[a_t](c)
end

function _in_first_col(c::CliffWorld, i::Int)
    return i <= c._rows
end

function _in_first_col(c::CliffWorld, col::Int, row::Int)
    return _in_first_col(c, _rowcol2index(c, col, row))
end

function _in_first_col(c::CliffWorld)
    return _in_first_col(c, c._current_state)
end

function _in_last_col(c::CliffWorld, i::Int)
    return (
        i > (c._rows * c._cols) - c._rows
    )
end

function _in_last_col(c::CliffWorld, col::Int, row::Int)
    return _in_last_col(c, _rowcol2index(c, col, row))
end

function _in_last_col(c::CliffWorld)
    return _in_last_col(c, c._current_state)
end

function _in_first_row(c::CliffWorld, i::Int)
    return mod(i, c._rows) == 1
end

function _in_first_row(c::CliffWorld, col::Int, row::Int)
    return _in_first_row(c, _rowcol2index(c, col, row))
end

function _in_first_row(c::CliffWorld)
    return _in_first_row(c, c._current_state)
end

function _in_last_row(c::CliffWorld, i::Int)
    return mod(i, c._rows) == 0
end

function _in_last_row(c::CliffWorld, col::Int, row::Int)
    return _in_last_row(c, _rowcol2index(c, col, row))
end

function _in_last_row(c::CliffWorld)
    return _in_last_row(c, c._current_state)
end
