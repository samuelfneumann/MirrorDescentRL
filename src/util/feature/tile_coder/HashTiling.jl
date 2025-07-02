# Todo: add offset, and use the integer-based version of tiles3, where a floating point
# vector is first converted to an integer, then the representation stored
#
# Add T offset
# Then convert data to an int by casting down or truncating
# Finally add that to the dict
#
# This file is incomplete

mutable struct HashTiling{IN<:Number,OUT<:Number} <: AbstractTiling{IN,OUT}
    size::Int
    overfull_count::Int
    dict::Dict{Vector{IN}, Int}
    hash
    tiling_number::Int
    offset::Vector{IN}

    function HashTiling{IN,OUT}(size::Integer, f=hash) where {IN<:Number,OUT}
        return new{IN,OUT}(size, 0, Dict{Vector{IN}, Int}(), f)
    end
end

function _count(h::HashTiling)
    return length(h.dict)
end

function index!(h::HashTiling, v::Vector, readonly=false)::Int
    v = v.+ offset

    if v in keys(h.dict)
        return h.dict[v]
    elseif readonly
        return -1
    end

    size = h.size
    count = _count(h)
    if count > size
        if h.overful_count == 0
            println("HashTiling full, starting to allow collisions")
        end
        h.overfull_count += 1

        return (h.hash(v) % h.size)::Int
    end
    h.dict[v] = count

    return count
end

function index!(h::HashTiling, v::Matrix, readonly=false)::Vector{Int}
    out::Vector{Int} = zeros(size(v)[1])
    for i in 1:size(v)[1]
        out[i] = index!(h, v[i, :])
    end

    return out
end

function onehot!(h::HashTiling{IN,OUT}, v::Vector)::Vector{OUT} where {IN<:Number,OUT}
    out = zeros(length(v))
    out[index!(h, v)] = 1

    return out
end
