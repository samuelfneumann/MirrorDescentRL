using LinearAlgebra

struct Linear{T<:Real,F}
    _in::Int
    _out::Int
    _init::F
end

Linear{T}(in, out; init::F=zeros) where {T,F} = Linear{T,F}(in, out, init)
Linear(T, in, out; init=zeros) = Linear{T}(in, out; init=init)
Linear(in, out; init=zeros) = Linear{Float32}(in, out; init=init)

function setup(rng::AbstractRNG, l::Linear{T}) where {T<:Real}
    if l._out == 1
        return (layer_1 = l._init(rng, T, l._in),), NamedTuple()
    else
        return (layer_1 = l._init(rng, T, l._in, l._out),), NamedTuple()
    end
end

function setup(l::Linear{T}) where {T<:Real}
    return setup(Random.GLOBAL_RNG, l)
end

(l::Linear)(x, ps, st) = predict(l, x, ps.layer_1, st)
predict(::Linear, x, ps::AbstractVector, st) = dot(ps, x)
predict(::Linear, x::AbstractVector, ps::AbstractMatrix, st) = (x' * ps)[1, :]
predict(::Linear, x::AbstractMatrix, ps::AbstractMatrix, st) = ps' * x

function predict(::Linear, x::AbstractVector{Bool}, ps::AbstractVector, st)
    return sum(ps[x]), st
end

function predict(::Linear, x::AbstractVector{Bool}, ps::AbstractMatrix, st)
    return sum(ps[x, :]; dims=1), st
end

function predict(::Linear, x::AbstractVector{Int}, ps::AbstractVector, st)
    return sum(view(ps, x)), st
end

function predict(::Linear, x::AbstractVector{Int}, ps::AbstractMatrix, st)
    return sum(view(ps, x, :); dims=1)[1, :], st
end

function predict(::Linear, x::AbstractMatrix{Int}, ps::AbstractMatrix, st)
    out = []
    for x_ in eachcol(x)
        push!(out, sum(view(ps, x_, :); dims=1)[1, :])
    end
    return stack(out), st
end

function predict(::Linear, x::Int, ps::AbstractVector, st)
    return ps[x], st
end

function predict(::Linear, x::Int, ps::AbstractMatrix, st)
    return ps[x, :], st
end

function setcol(::Linear, ind::Int, col::AbstractVector, ps::NamedTuple, st)
    ps = deepcopy(ps)
    ps.layer_1[:, ind] = col
    return ps
end

function setrow(::Linear, ind::Int, row::AbstractVector, ps::NamedTuple, st)
    ps = deepcopy(ps)
    ps.layer_1[ind, :] = row
    return ps
end
