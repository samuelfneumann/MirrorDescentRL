mutable struct ContinuousGridworld{
    A<:AbstractSpace,
    O<:AbstractSpace,
    R<:AbstractRNG,
} <: AbstractEnvironment
    _x::Float32
    _y::Float32
    _prev_x::Float32
    _prev_y::Float32

    _continuous::Bool

    _height::Float32
    _width::Float32

    _startxs::Vector{Float32}
    _startys::Vector{Float32}
    _goalxs::Vector{Float32}
    _goalys::Vector{Float32}
    _goal_radii::Vector{Float32}
    _goal_rewards::Vector{Float32}
    _timestep_reward::Float32

    _obsspace::O
    _actionspace::A
    _action_scale::Vector{Float32}

    _γ::Float32
    _rng::R

    function ContinuousGridworld(
        rng::R, γ, continuous, height, width, startxs, startys, goalxs, goalys, goal_radii,
        goal_rewards, timestep_reward, action_scale,
    ) where {R<:AbstractRNG}
        # Ensure valid height and width
        height <= 0f0 && error("height must be > 0")
        width <= 0f0 && error("width must be > 0")

        if goalxs === nothing
            goalxs = [width - goal_radii[1]]
        end
        if goalys ===  nothing
            goalys = [height - goal_radii[1]]
        end

        @assert length(goalxs) == length(goalys)
        @assert length(goalxs) == length(goal_rewards)
        @assert length(goalxs) == length(goal_radii)
        @assert length(startxs) == length(startys)
        @assert all(0f0 .<= startxs .<= width)
        @assert all(0f0 .<= startys .<= height)
        @assert all(0f0 .<= goalxs .<= width)
        @assert all(0f0 .<= goalys .<= height)

        low = Float32[0f0, 0f0]
        high = low .+ convert.(Float32, [width, height])
        obsspace = Box(low, high)
        O = typeof(obsspace)

        actionspace = if continuous
            action_max = Float32[1f0, 1f0]
            Box(-action_max, action_max)
        else
            Discrete(9)
        end
        A = typeof(actionspace)

        env = new{A,O,R}(
            0f0, 0f0, 0f0, 0f0, continuous, height, width, startxs, startys, goalxs, goalys,
            goal_radii, goal_rewards, timestep_reward, obsspace, actionspace, action_scale,
            γ, rng,
        )
        start!(env)
        return env
    end
end

function ContinuousGridworld(
    rng::AbstractRNG;
    continuous,
    γ,
    height = 1f0,
    width = 1f0,
    startxs = [5f-1],
    startys = [1f-1],
    goalxs = [1f-1, 9f-1],
    goalys = [3f-1, 9f-1],
    goal_radii=[0.1f0, 0.1f0],
    goal_rewards=[1f0, 10f0],
    timestep_reward=-0.01f0,
    action_scale=[1f-2, 1f-2],
)
    return ContinuousGridworld(
        rng, γ, continuous, height, width, startxs, startys, goalxs, goalys, goal_radii,
        goal_rewards, timestep_reward, action_scale,
    )
end

function start!(g::ContinuousGridworld)
    ind = rand(g._rng, UInt) % length(g._startxs) + 1
    g._x = g._startxs[ind]
    g._y = g._startys[ind]

    g._prev_x = g._startxs[ind]
    g._prev_y = g._startys[ind]

    return _obs(g)
end

function convert_to_continuous(g::ContinuousGridworld, action)
    return if g._continuous
        action
    else
        if only(action) == 1
            [1f0, 0f0]
        elseif only(action) == 2
            [0f0, 1f0]
        elseif only(action) == 3
            [-1f0, 0f0]
        elseif only(action) == 4
            [0f0, -1f0]
        elseif only(action) == 5
            [0f0, 0f0]
        elseif only(action) == 6
            [1f0, 1f0]
        elseif only(action) == 7
            [-1f0, 1f0]
        elseif only(action) == 8
            [1f0, -1f0]
        elseif only(action) == 9
            [-1f0, -1f0]
        end
    end
end

function apply_force(g::ContinuousGridworld, action)
    return action .* g._action_scale
end

function envstep!(g::ContinuousGridworld, action)
    check_contains_action(g, action)

    u = apply_force(g, convert_to_continuous(g, action))

    g._prev_x = g._x
    g._prev_y = g._y
    g._x = clamp(g._x + u[1], 0f0, g._width)
    g._y = clamp(g._y + u[2], 0f0, g._height)

    return _obs(g), reward(g), isterminal(g), γ(g)
end

function reward(g::ContinuousGridworld)
    goal = _get_goal_in(g)
    goal_reward = goal <= 0 ? 0f0 : g._goal_rewards[goal]
    return goal_reward + g._timestep_reward 
end

function _line_segment_in_circle(x1, y1, x2, y2, x_circle, y_circle, radius)
    # Calculates whether a line segment is inside a circle. The line segment is inside a
    # circle if either of the endpoints are inside the circle.
    point_in_circle = (x, y) -> (x - x_circle)^2 + (y - y_circle)^2 <= radius^2

    return point_in_circle(x1, y1) || point_in_circle(x2, y2)
end

function _line_segment_intersect_circle(x1, y1, x2, y2, x_circle, y_circle, radius)
    # Calculates the distance from the line segment from (x1, y1) to (x2, y2) to the circle
    # centre at (x_circle, y_circle) and checks if this distance is smaller than the circle
    # radius.
    #
    # See https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter

    ax = x1 - x_circle
    bx = x2 - x_circle
    ay = y1 - y_circle
    by = y2 - y_circle

    a = (bx - ax)^2 + (by - ay)^2
    b = 2 * (ax * (bx - ax) + ay * (by - ay))
    c = ax^2 + ay^2 - radius^2

    Δ = b^2 - 4 * a * c
    if Δ <= 0
        return false
    end

    t1 = (-b + sqrt(Δ)) / (2*a)
    t2 = (-b - sqrt(Δ)) / (2*a)

    return 0 < t1 < 1 || 0 < t2 < 1
end

function _in_goal(x1, y1, x2, y2, x_circle, y_circle, radius)
    # The agent has reached the goal if it is within the goal circle, or the path it took
    # from (x1, y1) to (x2, y2) touches the goal circle
    return (
        _line_segment_intersect_circle(x1, y1, x2, y2, x_circle, y_circle, radius) ||
        _line_segment_in_circle(x1, y1, x2, y2, x_circle, y_circle, radius)
    )
end

function _get_goal_in(g::ContinuousGridworld)
    for i in eachindex(g._goalxs)
        x0, y0, r0 = g._goalxs[i], g._goalys[i], g._goal_radii[i]
        x1, y1 = g._prev_x, g._prev_y
        x2, y2 = g._x, g._y
        if _in_goal(x1, y1, x2, y2, x0, y0, r0)
            return i
        end
    end
    return -1
end

function _in_any_goal(g::ContinuousGridworld)
    for i in eachindex(g._goalxs)
        x0, y0, r0 = g._goalxs[i], g._goalys[i], g._goal_radii[i]
        x1, y1 = g._prev_x, g._prev_y
        x2, y2 = g._x, g._y
        if _in_goal(x1, y1, x2, y2, x0, y0, r0)
            return true
        end
    end
    return false
end

isterminal(g::ContinuousGridworld)::Bool = _in_any_goal(g)
γ(g::ContinuousGridworld) = isterminal(g) ? 0f0 : g._γ
observation_space(g::ContinuousGridworld) = g._obsspace
action_space(g::ContinuousGridworld) = g._actionspace
_obs(g::ContinuousGridworld) = [g._x, g._y]

function Base.show(io::IO, g::ContinuousGridworld)
    println(io, "ContinuousContinuousGridworld")
end
