####################################################################
# Initialization Functions
####################################################################
function uniform(::AbstractRNG, T, n_actions::Int, n_states::Int)
    probs = Base.ones(T, n_actions) ./ n_actions
    return repeat(probs, 1, n_states)
end

function uniform(rng::AbstractRNG, n_actions::Int, n_states::Int)
    return uniform(rng, Float32, n_actions, n_states)
end

function default_cliffworld_to_cliff(::AbstractRNG, T, n_actions::Int, n_states::Int)
    probs = zeros(T, n_actions, n_states)
    probs[2, :] .= 1
    probs[4, 44:48] .= 1
    probs[2, 44:48] .= 0
    probs[:, 1:4] .= 0.25
    # probs[3, 1:4] .= 1
    # probs[2, 1:4] .= 0

    return probs
end

# Trick to make initializing with Base.zeros and Base.ones work similarly to initializing
# with Lux's weight initializers
Base.zeros(rng, T, args...) = zeros(T, args...)
Base.ones(rng, T, args...) = ones(T, args...)
####################################################################

####################################################################
# Models Utilities
####################################################################
const Model = Union{Tabular,Linear,LuxModel}

function setup(seed::Integer, model::Model)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    return setup(rng, model)
end

train(::Model, model_st::NamedTuple) = model_st
eval(::Model, model_st::NamedTuple) = model_st

macro treemap(f, args...)
    :(treemap($f, $args...))
end

function treemap(f::Function, args...)
    return _treemap(f, args..., NamedTuple())
end

function _treemap(f::Function, n, out::NamedTuple)
    for key in keys(n)
        if n[key] isa NamedTuple
            v = (_treemap(f, deepcopy(n[key]), NamedTuple()),)
            k = (key,)
            out = merge(out, NamedTuple{k}(v))
        else
            k = (key,)
            v = (f(deepcopy(n[key])),)
            out = merge(out, NamedTuple{k}(v))
        end
    end
    return out
end

function _treemap(f::Function, n, m, out::NamedTuple)
    for key in keys(n)
        if n[key] isa NamedTuple
            v = (_treemap(f, deepcopy(n[key]), deepcopy(m[key]), NamedTuple()),)
            k = (key,)
            out = merge(out, NamedTuple{k}(v))
        else
            k = (key,)
            v = (f(deepcopy(n[key]), deepcopy(m[key])),)
            out = merge(out, NamedTuple{k}(v))
        end
    end
    return out
end

function _treemap(f::Function, n, m, p, out::NamedTuple)
    for key in keys(n)
        if n[key] isa NamedTuple
            v = (
                _treemap(
                    f, deepcopy(n[key]), deepcopy(m[key]), deepcopy(p[key]), NamedTuple(),
                ),
            )
            k = (key,)
            out = merge(out, NamedTuple{k}(v))
        else
            k = (key,)
            v = (f(deepcopy(n[key]), deepcopy(m[key]), deepcopy(p[key])),)
            out = merge(out, NamedTuple{k}(v))
        end
    end
    return out
end

macro treemap!(f, args...)
    :(treemap!($f, $args...))
end

function treemap!(f::Function, args...)
    return _treemap!(f, args..., NamedTuple())
end

function _treemap!(f::Function, n, out::NamedTuple)
    for key in keys(n)
        if n[key] isa NamedTuple
            v = (_treemap!(f, n[key], NamedTuple()),)
            k = (key,)
            out = merge(out, NamedTuple{k}(v))
        else
            k = (key,)
            v = (f(n[key]),)
            out = merge(out, NamedTuple{k}(v))
        end
    end
    return out
end

function _treemap!(f::Function, n, m, out::NamedTuple)
    for key in keys(n)
        if n[key] isa NamedTuple
            v = (_treemap!(f, n[key], m[key], NamedTuple()),)
            k = (key,)
            out = merge(out, NamedTuple{k}(v))
        else
            k = (key,)
            v = (f(n[key], m[key]),)
            out = merge(out, NamedTuple{k}(v))
        end
    end
    return out
end

function _treemap!(f::Function, n, m, p, out::NamedTuple)
    for key in keys(n)
        if n[key] isa NamedTuple
            v = (
                _treemap!(
                    f, n[key], m[key], p[key], NamedTuple(),
                ),
            )
            k = (key,)
            out = merge(out, NamedTuple{k}(v))
        else
            k = (key,)
            v = (f(n[key], m[key], p[key]),)
            out = merge(out, NamedTuple{k}(v))
        end
    end
    return out
end

####################################################################
