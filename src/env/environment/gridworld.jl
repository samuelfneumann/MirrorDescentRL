"""
    Gridworld <: AbstractEnvironment

A gridworld environment with a number of starting and goal states.

# Description
In this tabular environment, the agent will start in some (x, y)
position - possibly randomly from the set of all starting positions.
The agent can move left, right, up, and down. Actions that would take
the agent off the grid leave the agent in place. Certain states are
terminal/goal states. Upon entering one of these states, the episode
ends.

# State and Observations
The state/observations returned by the Gridworld can be either onehot
encodings of the (x, y) position of the agent in the gridworld,
the (x, y) positions themselves, or an index. Gridworlds are tabular environments,
and so the onehot encoding determines exactly where the agent is in
the environment, and is a somewhat easier problem.

# Actions
Actions are discrete in the set (1, 2, 3, 4) and have the following
interpretations:

Action | Meaning
-------|------------------------
   1   | Move up
   2   | Move right
   3   | Move down
   4   | Move left

# Goal/Rewards
The goal of the agent is to end the episode as quickly as possible by
entering a terminal/goal state. A reward of -1 is given on all
transitions, except the transition to a goal state, when a reward of 0
is given.

# Fields
- `obsspace::AbstractSpace`: the observation space
- `actionspace::AbstractSpace`: the action space
- `rows::Int`:: see constructor
- `cols::Int`:: see constructor
"""
mutable struct Gridworld{
        S<:Union{AbstractFloat, Integer},  # State observation type: AbstractArray{S}
        A<:AbstractSpace,
        O<:AbstractSpace{S},
        R<:AbstractRNG,
} <: AbstractEnvironment
    _x::Int
    _y::Int
    _rows::Int
    _cols::Int

    _startxs::Vector{Int}
    _startys::Vector{Int}
    _goalxs::Vector{Int}
    _goalys::Vector{Int}

    _obsspace::O
    _actionspace::A

    _goal_reward::Float32
    _timestep_reward::Float32

    _γ::Float32
    _rng::R
    _state_repr::Symbol

    _exploring_starts::Bool

    function Gridworld(
        x,
        y,
        rows,
        cols,

        startxs,
        startys,
        goalxs,
        goalys,

        obsspace::O,
        actionspace::A,

        goal_reward,
        timestep_reward,

        γ,
        rng::R,
        state_repr,
    ) where {A,S,O<:AbstractSpace{S},R}
        @assert state_repr in (:onehot, :index, :coordinates)

        if S !== eltype(eltype(obsspace))
            T = eltype(eltype(obsspace))
            error("expected state observation to be $T but got $S")
        end

        exploring_starts = (
            length(startys |> unique) == rows && length(startxs |> unique) == cols 
        )

        return new{S,A,O,R}(
            x,
            y,
            rows,
            cols,

            startxs,
            startys,
            goalxs,
            goalys,

            obsspace,
            actionspace,

            goal_reward,
            timestep_reward,

            γ,
            rng,
            state_repr,
            exploring_starts,
        )
    end
end

"""
    Gridworld(rng::AbstractRNG; kwargs...)

Constructor for the gridworld environment.

The gridworld may have many starting positions, determined by `startxs` and `startys`
respectively. These two vectors must have the same length. The starting position is
determine randomly from `(startxs[i], startys[i])` every time `start!` is called.

Similarly, multiple goal positions can be specified with `goalxs` and `goalys`, which also
must be of the same length.
"""
function Gridworld(
    rng::AbstractRNG;
    rows = 10,
    cols = 10,
    startxs = [1],
    startys = [1],
    goalxs = nothing,
    goalys = nothing,
    goal_reward = 0f0,
    timestep_reward = -1f0,
    γ = 1.0f0,
    state_repr = :index,
    use_floating_point_obs = false,
    exploring_starts = false,
)
    # Ensure the number of rows and columns are more than 1
    rows < 1 && error("rows must be larger than 1")
    cols < 1 && error("cols must be larger than 1")

    if goalxs === nothing
        goalxs = [cols]
    end
    if goalys ===  nothing
        goalys = [rows]
    end

    if exploring_starts
        if startxs != [1] || startys != [1]
            @warn "specificy `exploring_Starts`: replacing startxs and startys"
        end
        startxs = []
        startys = []
        for c in collect(1:cols)
            for r in collect(1:rows)
                push!(startxs, c)
                push!(startys, r)
            end
        end
    end

    # Check to ensure that the start positions are legal
    if length(startxs) != length(startys)
        error("start x positions should have the same length as start y positions")
    end
    for i = 1:length(startxs)
        startx = startxs[i]
        starty = startys[i]
        startx < 1 && error("startx must be larger than 1")
        startx > cols && error("startx must not exceed cols ($cols)")
        starty < 1 && error("starty must be larger than 1")
        starty > rows && error("starty must not exceed rows ($rows)")
    end

    # Check to ensure that the goal positions are legal
    if length(goalxs) != length(goalys)
        error("goal x positions should have the same length as goal y positions")
    end
    for i = 1:length(goalxs)
        goalx = goalxs[i]
        goaly = goalys[i]
        goalx < 1 && error("goalx must be larger than 1")
        goalx > cols && error("goalx must not exceed cols ($cols)")
        goaly < 1 && error("goaly must be larger than 1")
        goaly > rows && error("goaly must not exceed rows ($rows)")
    end

    # Create observation and action spaces
    low = zeros(rows * cols)
    high = ones(rows * cols)

    S = use_floating_point_obs ? Float32 : Int
    obsspace = if state_repr == :index
        Discrete(rows * cols)
    else
        Box{S}(low, high)
    end

    actionspace = Discrete(4)

    g = Gridworld(
        0,
        0,
        rows,
        cols,
        startxs,
        startys,
        goalxs,
        goalys,
        obsspace,
        actionspace,
        goal_reward,
        timestep_reward,
        γ,
        rng,
        state_repr,
    )
    start!(g)
    return g
end

function start!(g::Gridworld{S})::AbstractArray{S} where {S<:Number}
    ind = rand(g._rng, UInt) % length(g._startxs) + 1
    g._x = g._startxs[ind]
    g._y = g._startys[ind]

    return _obs(g)
end

function dynamics_matrix(
    g::Gridworld, π, π_f, π_θ, π_st,
)::Matrix{Float32}
    size = length(g)
    P = zeros(Float32, (size, size))

    for s1_ind in 1:size
        s1_xy = _index2xy(g, s1_ind)
        obs = _obs(g, s1_xy)
        action_probs, _ = prob(π, π_f, π_θ, π_st, obs)
        s2_ind = get_next_states_ind(g, s1_xy)

        for (i, state_ind) in enumerate(s2_ind)
            P[s1_ind, state_ind] += action_probs[i]
        end
    end

    return P
end

function discounted_visitation(
    g::Gridworld, π, π_f, π_θ, π_st, ρ::AbstractVector{Float32},
)::Vector{Float32}
    @assert size(ρ, 1) == length(g)
    P = dynamics_matrix(g, π, π_f, π_θ, π_st)
    d = (1 - g._γ) .* inv(I(length(g)) .- (g._γ .* P))
    return dropdims(collect(ρ' * d); dims=1)
end

function discounted_visitation(
    g::Gridworld, π, π_f, π_θ, π_st,
)::Vector{Float32}
    s = zeros(Float32, length(g))

    ρ = if g._exploring_starts
        ones(Float32, length(g)) ./ length(g)
    else
        # Construct restart distribution
        for start_x in g._startxs
            for start_y in g._startys
                s[_xy2index(g, (x=start_x, y=start_y))] += 1
            end
        end
        s / sum(s)
    end

    return discounted_visitation(g, π, π_f, π_θ, π_st, ρ)
end

"""
Returns the next states for taking actions, 1, 2, 3, and 4 in that order from state
`current_state`
"""
function get_next_states_xy(
    g::Gridworld,
    current_state::NamedTuple{(:x, :y), Tuple{Int, Int}},
)
    return [
        (x = current_state.x, y=max(1, current_state.y - 1)),
        (x = min(g._cols, current_state.x + 1), y=current_state.y),
        (x = current_state.x, y=min(g._rows, current_state.y + 1)),
        (x = max(1, current_state.x - 1), y=current_state.y),
    ]
end

function get_next_states_ind(
    g::Gridworld, current_state::NamedTuple{(:x, :y), Tuple{Int, Int}},
)
    return _xy2index.([g], get_next_states_xy(g, current_state))
end


function envstep!(g::Gridworld, action)
    check_contains_action(g, action)

    if action isa AbstractArray
        action = action[1]
    end

    if action == 1 && g._y > 1
        # Move down
        g._y -= 1
    elseif action == 2 && g._x < g._cols
        # Move right
        g._x += 1
    elseif action == 3 && g._y < g._rows
        # Move up
        g._y += 1
    elseif action == 4 && g._x > 1
        # Move left
        g._x -= 1
    end

    return _obs(g), reward(g), isterminal(g), γ(g)
end

function reward(g::Gridworld)
    return isterminal(g) ? g._goal_reward : g._timestep_reward
end

function isterminal(g::Gridworld)::Bool
    return g._x in g._goalxs && g._y in g._goalys
end

function γ(g::Gridworld)
    isterminal(g) ? 0f0 : g._γ
end

function observation_space(g::Gridworld)
    return g._obsspace
end

function action_space(g::Gridworld)::AbstractSpace{Int,1}
    return g._actionspace
end

function _obs(g::Gridworld{S}, state::NamedTuple{(:x, :y), Tuple{Int, Int}}) where {S<:Number}
    return if g._state_repr == :coordinates
        [state.x, state.y]
    elseif g._state_repr == :onehot
        obs = zeros(S, length(g))
        obs[_xy2index(g, state)] = one(S)
        obs
    else
        [_xy2index(g, state)]
    end
end

function _obs(g::Gridworld{S}) where {S<:Number}
    _obs(g, state_tuple(g))
end

state_tuple(g::Gridworld) = (x = g._x, y = g._y)

function _current_index(g::Gridworld)::Int
    return _xy2index(g, state_tuple(g))
end

function _index2xy(g::Gridworld, index::Integer)
    x = (index - 1) ÷ g._rows
    y = index - (x * g._rows)
    return (x=x+1, y=y)
end

function _xy2index(g::Gridworld, state::NamedTuple{(:x, :y), Tuple{Int, Int}})::Int
    return (state.x - 1) * g._rows + state.y
end


function Base.length(g::Gridworld)
    return g._rows * g._cols
end

function Base.show(io::IO, g::Gridworld)
    println(io, "Gridworld")
end
