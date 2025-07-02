# Tabular only supports int or bool observations
struct Tabular{T<:Real,F}
    _n_states::Int
    _n_actions::Int
    _init::F
end

# In = num states, out = num actions
function Tabular{T}(n_states, n_actions; init::F=zeros) where {T,F}
    Tabular{T,F}(n_states, n_actions, init)
end

Tabular(T, n_states, n_actions; init=zeros) = Tabular{T}(n_states, n_actions; init=init)
Tabular(n_states, n_actions; init=zeros) = Tabular{Float32}(n_states, n_actions; init=init)

function setup(rng::AbstractRNG, t::Tabular{T}) where {T<:Real}
    return (layer_1 = t._init(rng, T, t._n_actions, t._n_states),), NamedTuple()
end

function setup(t::Tabular{T}) where {T<:Real}
    return setup(Random.GLOBAL_RNG, t)
end

function predict(t::Tabular, x, ps::NamedTuple{(:layer_1,), Tuple{T}}, st) where {T}
    @assert size(ps.layer_1) == (t._n_actions, t._n_states)
    predict(t, x, ps.layer_1, st)
end

function predict(t::Tabular, ps::NamedTuple{(:layer_1,), Tuple{T}}, st) where {T}
    @assert size(ps.layer_1) == (t._n_actions, t._n_states)
    predict(t, ps.layer_1, st)
end

function (t::Tabular)(args...)
    predict(t, args...)
end

function predict(::Tabular, ps, st)
    return deepcopy(ps), st
end

function predict(::Tabular, x::AbstractVector{Bool}, ps::AbstractVector, st)
    # return sum(ps[x]), st
    return sum(view(ps, x)), st
end

# Select all actions in some states
function predict(::Tabular, x::AbstractVector{Bool}, ps::AbstractMatrix, st)
    out = ps[:, x]
    out = if size(out, 2) == 1
        dropdims(out; dims=2)
    else
        out
    end
    return out, st
end

function predict(::Tabular, x::AbstractVector{Int}, ps::AbstractVector, st)
    return ps[x], st
end

# Select all actions in some states
function predict(::Tabular, x::AbstractVector{Int}, ps::AbstractMatrix, st)
    out = ps[:, x]
    out = if size(out, 2) == 1
        dropdims(out; dims=2)
    else
        out
    end
    return out, st
end

# Select all actions in some states
function predict(::Tabular, x::AbstractMatrix{Int}, ps::AbstractMatrix, st)
    out = [ps[:, x_][:, 1] for x_ in eachcol(x)]
    return reduce(hcat, out), st
end

function predict(::Tabular, x::Int, ps::AbstractVector, st)
    return ps[x], st
end

# Select all actions in some states x
function predict(::Tabular, x::Int, ps::AbstractMatrix, st)
    return ps[:, x], st
end

function setcol(::Tabular, ind::Int, col::AbstractVector, ps::NamedTuple)
    ps = deepcopy(ps)
    ps.layer_1[:, ind] = col
    return ps
end

function setcol(::Tabular, ind::Int, col::AbstractMatrix, ps::NamedTuple)
    @assert size(col, 2) == 1
    ps = deepcopy(ps)
    ps.layer_1[:, ind] = col[:, 1]
    return ps
end

function setrow(::Tabular, ind::Int, row::AbstractVector, ps::NamedTuple)
    ps = deepcopy(ps)
    ps.layer_1[ind, :] = row
    return ps
end

function set(::Tabular, ps::NamedTuple, val)
    return (layer_1 = val,)
end

function set(::Tabular, ps::NamedTuple, row::Int, col::Int, val)
    ps = deepcopy(ps)
    ps.layer_1[row, col] = val
    return ps
end

function set(::Tabular, ps::NamedTuple, row::Vector{Int}, col::Vector{Int}, val)
    ps = deepcopy(ps)
    ps.layer_1[CartesianIndex.(row, col)] .= val
    return ps
end

Base.eltype(::Tabular{T}) where {T} = T
Base.zero(T::Type, t::Tabular) = (layer_1 = zeros(T, t._n_actions, t._n_states),)
Base.zero(t::Tabular) = zero(eltype(t), t)
Base.size(t::Tabular) = (t._n_actions, t._n_states)
Base.size(t::Tabular, i::Int) = (t._n_actions, t._n_states)[i]
SparseArrays.spzeros(t::Tabular) = spzeros(eltype(t), t)

function SparseArrays.spzeros(T::Type, t::Tabular)
    (layer_1 = spzeros(T, t._n_actions, t._n_states),)
end
