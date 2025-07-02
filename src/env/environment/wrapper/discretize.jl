"""
    Discretize{P1,P2} <: AbstractEnvironmentActionWrapper where {P1<:Integer,P2<:AbstractFloat}

Discretize a 1-dimensional continuous actions space.

The `P1` type parameter determines the precision of integers to use and defaults to
`Int`. The `P2` type parameter should be the type of continuous actions the
wrapped environment expects and is set automatically by the constructor.
"""
struct Discretize{P1,P2} <: AbstractEnvironmentActionWrapper where {P1<:Integer,P2<:AbstractFloat}
    _env::AbstractEnvironment
    _action_space::AbstractSpace

    _action_map::Dict{P1,Vector{P2}}

    function Discretize{P1}(env::AbstractEnvironment, actions::Integer) where {P1<:Integer}
        if ! continuous(action_space(env))
            error("cannot discretize environment with non-continuous action space")
        elseif actions <= 0
            error("actions must be larger than 0")
        elseif length(size(action_space(env))) != 1 || size(action_space(env))[1] != 1
            error("can only discretize a 1-dimensional action space")
        end

        P = typeof(low(action_space(env))[1])
        action_map = Dict{P1,Vector{P}}()

        lower = low(action_space(env))
        upper = high(action_space(env))

        if actions > 1
            step = (upper - lower) ./ (actions-1)

            for action in 1:actions
                action_map[action] = (action - 1) * step .+ lower
            end
        else
            action_map[actions] = (upper - lower) ./ 2
        end

        return new{P1,P}(env, Discrete{P1}(actions), action_map)
    end
end

"""
    Discretize{P}(env::AbstractEnvironment, actions::Integer) where {P<:Integer}
    Discretize(env::AbstractEnvironment, actions::Integer)

Construct a `Discretize` struct.

The type parameter `P` determines the type of integer actions the wrapper expects. If it is
left unspecified, then `Int` is used.

# Arguments
- `env::AbstractEnvironment`: the environment whose action space to discretize; must be a
continuous action environment
- `actions::Integer`: the number of discrete actions to use in the environment
"""
function Discretize(env::AbstractEnvironment, actions::Integer)
    return Discretize{Int}(env, actions)
end

function action_space(d::Discretize)::AbstractSpace
    return d._action_space
end

function action(d::Discretize, a)
    if ! contains(action_space(d), a)
        throw(DomainError(a, "action $a ∉ action space $(action_space(d))"))
    end

    return d._action_map[a]
end

function wrapped(d::Discretize)::AbstractEnvironment
    return d._env
end
