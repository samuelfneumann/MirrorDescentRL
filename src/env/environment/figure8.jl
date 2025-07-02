mutable struct Figure8{
    A<:AbstractSpace,
    O<:AbstractSpace,
    R<:AbstractRNG,
} <: AbstractEnvironment

    _action_space::A
    _observation_space::O

    _dt::Float32
    _minSteps::Int
    _maxSteps::Int

    _γ::Float32

    t::Float32
    tg::Float32
    x::Float32

    function Figure8(
        rng::R, continuous, loops, ε=3f-1,γ,
    ) where {R<:AbstractRNG}

        minSteps, maxSteps = if loops !== nothing && loops > 0
            maxT = loops * 2π + dt
            floor(1//2 * ε / dt), floor(maxT / dt)
        else
            -1, -1
        end

        action_space = if continuous
            error("continuous-action Figure8 not implemented")
        else
            Discrete(9)
        end
        A = typeof(action_space)

        # TODO: observation space
        observation_space = nothing
        O = typeof(observation_space)

        a = new{A,O,R}(
            action_space, observation_space, 0.1f0, minSteps, maxSteps, γ, 0f0, 0f0, 0f0,
        )

        start!(a)
        return a
    end
end

function Figure8(rng::AbstractRNG; continuous, γ=1.0f0)
    return Figure8(rng, continuous, sparse_rewards, γ)
end

function start!(a::Figure8)
    t = 0f0
    tg = 0f0
    x = zeros(Float32, 4)

    p = _figure8_pattern(tg)
    x[1] = p[1]
    x[2] = p[2]

    goalp = _figure8_pattern(tg + dt)
    x[3] = goalp[1]
    x[4] = goalp[2]

    obs = @view x[3:4]
    a._t, a._tg, a.x = (t, tg, x)
    return obs
end

function envstep!(a::Figure8, action)
    check_contains_action(a, action)
    # TODO
    _figure8_step(s, a, dt, maxT, ε, γ)
    return a._state, reward(a), isterminal(a), γ(a)
end

function _figure8_discrete2contdiag(a::Int)
    if a == 1
        return (0.0, 1.0)
    elseif a == 2
        return (1.0, 0.0)
    elseif a == 3
        return (0.0, -1.0)
    elseif a == 4
        return (-1.0, 0.0)
    elseif a == 5
        return (1.0, 1.0)
    elseif a == 6
        return (1.0, -1.0)
    elseif a == 7
        return (-1.0, -1.0)
    elseif a == 8
        return (-1.0, 1.0)
    else
        return (0.0, 0.0)
    end
end

function _figure8_step(s, a, dt, maxT, ϵ, γ=1f0)
    t, tg, x = s
    Δp = _figure8_discrete2contdiag(a)
    t += dt
    tg += dt

    p = @view x[1:2]
    p .= p .+ dt .* Δp
    p[1] = clamp(p[1], -1.2f0, 1.2f0)
    p[2] = clamp(p[2], -0.6f0, 0.6f0)

    goalp = _figure8_pattern(tg)
    d = _figure8_dist(p, goalp)

    # This is the is_terminal condition
    if d > ϵ || t ≥ (maxT - 1f-8)
        γ = 0f0
    end
    r = 1f0

    goalp = figure8_pattern(t+dt)
    x[3] = goalp[1] - p[1]
    x[4] = goalp[2] - p[2]

    s′ = (t, tg, x) # TODO: this should be part of the env state
    obs = @view x[3:4]

    return obs, r, γ
end

function reward(a::Figure8)
    if a._sparse_rewards
    end
end

function isterminal(a::Figure8)::Bool
end

function observation_space(a::Figure8)
    a._observationspace
end

function action_space(a::Figure8)
    a._actionspace
end

function γ(a::Figure8)
    isterminal(a) ? 0f0 : a._γ
end

function Base.show(io::IO, a::Figure8)
    print(io, "Figure8")
end

function _figure8_pattern(t)
    x = sin(t)
    y = sin(t) * cos(t)
    return x, y
end


function _figure8_dist(p, g)
    d = (p[1] - g[1])^2 + (p[2] - g[2])^2
    return sqrt(d)
end
