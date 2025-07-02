const ALIASED_STATE_N_STATES = 10

struct IntNode{N}
    _value::Int
    _children::NTuple{N,IntNode}
end

IntNode(v) = IntNode(v, tuple())
IntNode(v, c::Vector) = IntNode(v, tuple(c...))

AbstractTrees.nodevalue(i::IntNode) = i._value
AbstractTrees.children(i::IntNode) = i._children
AbstractTrees.NodeType(::Type{IntNode}) = HasNodeType()
AbstractTrees.nodetype(::Type{IntNode}) = IntNode

mutable struct AliasedState{
    T<:Real,
    A<:AbstractSpace,
    O<:AbstractSpace,
    R<:AbstractRNG,
} <: AbstractEnvironment
    _rng::R
    _observationspace::O
    _actionspace::A
    _γ::Float32
    _current_step::Int
    _last_action::Int
    _int_obs::Bool
    _tree::IntNode
    _current_state::IntNode

    _r::Float32
    _top_r1::Float32
    _top_r2::Float32
    _ε::Float32

    function AliasedState{T}(rng::R, γ, int_obs) where {T<:Real,R<:AbstractRNG}

        if int_obs
            obs_space = Discrete{Int}(ALIASED_STATE_N_STATES - 1)
        else
            low = zeros(T, ALIASED_STATE_N_STATES - 1)
            high = ones(T, ALIASED_STATE_N_STATES - 1)
            obs_space = Box{Int}(low, high)
        end

        action_space = Discrete(2)
        O = typeof(obs_space)
        A = typeof(action_space)

        tree = IntNode(
            1, [
                IntNode(2, [IntNode(5), IntNode(6)]),
                IntNode(3, [IntNode(7), IntNode(8)]),
                IntNode(4, [IntNode(9), IntNode(10)]),
            ]
        )

        top_r1 = -1f0
        top_r2 = -100f0
        r = -10f0
        ε = ((r - top_r1) / (top_r2 - top_r1)) / 25f0

        p = new{T,A,O,R}(
            rng, obs_space, action_space, γ, 1, -1, int_obs, tree, tree, r, top_r1, top_r2,
            ε,
        )

        start!(p)
        return p
    end
end

function AliasedState(rng::AbstractRNG; γ=1f0, int_obs=true)
    AliasedState{Int}(rng; γ=γ, int_obs=int_obs)
end

function AliasedState{T}(rng; γ=1f0, int_obs=true) where {T<:Real}
    AliasedState{T}(rng, γ, int_obs)
end

function reward(a::AliasedState)
    if a._last_action < 1
        @warn "should not call reward unless action has been taken" maxlog=1
        return 0f0
    end

    return if nodevalue(a._current_state) == 5 && a._last_action == 1
        a._top_r1
    elseif nodevalue(a._current_state) == 7 && a._last_action == 1
        a._top_r2
    # elseif nodevalue(a._current_state) == 9 && a._last_action == 1
    #     a._top_r1 * 2
    else
        a._r
    end

end

is_aliased_state(a::AliasedState) = nodevalue(a._current_state) in (2, 3)
γ(a::AliasedState) = isterminal(a) ? zero(a._γ) : a._γ
observation_space(a::AliasedState) = a._observationspace
action_space(a::AliasedState) = a._actionspace
isterminal(a::AliasedState) = _at_goal(a)
_at_goal(a::AliasedState) = nodevalue(a._current_state) in (5, 6, 7, 8, 9, 10)

function start!(a::AliasedState{T}) where {T}
    a._current_state = a._tree
    a._last_action = -1

    return _get_obs(a)
end

function envstep!(a::AliasedState, action)
    check_contains_action(a, action)

    @assert length(action) == 1 # actions are integers
    action = action[1]
    @assert action == 1 || action == 2 "expected action ∈ {1, 2}"
    a._current_step += 1
    a._last_action = action

    if nodevalue(a._current_state) == 1 && action == 1
        p = rand(a._rng)
        if p < a._ε
            a._current_state = children(a._current_state)[2]
        else
            a._current_state = children(a._current_state)[1]
        end
    elseif nodevalue(a._current_state) == 1 && action == 2
        a._current_state = children(a._current_state)[3]
    else
        a._current_state = children(a._current_state)[action]
    end

    return _get_obs(a), reward(a), isterminal(a), γ(a)
end

function _get_obs(a::AliasedState{T}) where {T}
    value = nodevalue(a._current_state)
    value = value > 2 ? value - 1 : value
    return if a._int_obs
        [value]
    else
        z = spzeros(T, ALIASED_STATE_N_STATES)
        z[value] = one(T)
        z
    end
end

function Base.show(io::IO, a::AliasedState{T}) where {T}
    s1 = """
            O
        O _ |
       |    O
       |
       |    O
       -O _ |
       |    O
    x -
       |    O
        O - |
            O
    """

    s2 = """
            O
        x _ |
       |    O
       |
       |    O
       -O _ |
       |    O
    O -
       |    O
        O - |
            O
    """

    s3 = """
            O
        O _ |
       |    O
       |
       |    O
       -x _ |
       |    O
    O -
       |    O
        O - |
            O
    """

    s4 = """
            O
        O _ |
       |    O
       |
       |    O
       -O _ |
       |    O
    O -
       |    O
        x - |
            O
    """

    s5 = """
            x
        O _ |
       |    O
       |
       |    O
       -O _ |
       |    O
    O -
       |    O
        O - |
            O
    """

    s6 = """
            O
        O _ |
       |    x
       |
       |    O
       -O _ |
       |    O
    O -
       |    O
        O - |
            O
    """

    s7 = """
            O
        O _ |
       |    O
       |
       |    x
       -O _ |
       |    O
    O -
       |    O
        O - |
            O
    """

    s8 = """
            O
        O _ |
       |    O
       |
       |    O
       -O _ |
       |    x
    O -
       |    O
        O - |
            O
    """

    s9 = """
            O
        O _ |
       |    O
       |
       |    O
       -O _ |
       |    O
    O -
       |    x
        O - |
            O
    """

    s10 = """
            O
        O _ |
       |    O
       |
       |    O
       -O _ |
       |    O
    O -
       |    O
        O - |
            x
    """

    print(io, [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10][nodevalue(a._current_state)])
    println(io, "AliasedState{$T}")
end

