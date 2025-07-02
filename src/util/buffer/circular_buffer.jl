"""
    CircularBuffer{TPL, TPS, NAMES}

Maintains a buffer of fixed size w/o reallocating and deallocating memory through a
circular queue data struct.
"""
mutable struct CircularBuffer{TPL, TPS, NAMES}
    """The structure the data is stored"""
    _stg_tuple::TPL

    """Current column."""
    _current_row::Int64

    """Max size"""
    _capacity::Int64

    """Whether the datastruct is full (i.e. has gone through a single rotation)"""
    _full::Bool

    """Data_types of the data stored in the buffer."""
    _data_types::TPS
    function CircularBuffer(size, types, shapes, column_names)
        @assert length(types) == length(shapes)
        @assert length(types) == length(column_names)

        data = NamedTuple{column_names}(
            shapes[i] == 1 ? zeros(types[i], size) : zeros(types[i], shapes[i]..., size)
            for i in 1:length(types)
        )
        return new{typeof(data), typeof(types), column_names}(data, 1, size, false, types)
    end

    function CircularBuffer{TPL,TPS,NAMES}(
        stg_tuple::TPL,
        current_row,
        capacity,
        full,
        data_types::TPS,
    ) where {TPL,TPS,NAMES}
        return new{TPL,TPS,NAMES}(stg_tuple, current_row, capacity, full, data_types)
    end
end

function CircularBuffer(
    stg_tuple::TPL, current_row, capacity, full, data_types::TPS,
) where {TPL,TPS}
    NAMES = keys(stg_tuple)
    return CircularBuffer{TPL,TPS,NAMES}(stg_tuple, current_row, capacity, full, data_types)
end

Adapt.@adapt_structure CircularBuffer

full(cb::CircularBuffer) = cb._full

function reset!(cb::CircularBuffer)
    cb._full = false
    cb._current_row = 1
end

_get_data_row(x::AbstractArray{T, 1}, idx) where {T} = x[idx]
_get_data_row(x::AbstractArray{T, 2}, idx) where {T} = x[:, idx]
_get_data_row(x::AbstractArray{T, 3}, idx) where {T} = x[:, :, idx]
_get_data_row(x::AbstractArray{T, 4}, idx) where {T} = x[:, :, :, idx]
_get_data_row(x::AbstractArray{T, 5}, idx) where {T} = x[:, :, :, :, idx]

_get_data_row_view(x::AbstractArray{T, 1}, idx) where {T} = @view x[idx]
_get_data_row_view(x::AbstractArray{T, 2}, idx) where {T} = @view x[:, idx]
_get_data_row_view(x::AbstractArray{T, 3}, idx) where {T} = @view x[:, :, idx]
_get_data_row_view(x::AbstractArray{T, 4}, idx) where {T} = @view x[:, :, :, idx]
_get_data_row_view(x::AbstractArray{T, 5}, idx) where {T} = @view x[:, :, :, :, idx]

_set_data_row!(x::AbstractArray{T, 1}, d::T, idx) where {T} = x[idx] = d
_set_data_row!(x::AbstractArray{T, 1}, d::AbstractArray{T, 1}, idx) where {T} = x[idx] = d[1]
_set_data_row!(x::AbstractArray{T, 2}, d::AbstractArray{T, 1}, idx) where {T} = x[:, idx] .= d
_set_data_row!(x::AbstractArray{T, 2}, d::AbstractArray{T, 2}, idx) where {T} = x[:, idx] .= reshape(d, :)
_set_data_row!(x::AbstractArray{T, 3}, d::AbstractArray{T, 2}, idx) where {T} = x[:, :, idx] .= d
_set_data_row!(x::AbstractArray{T, 4}, d::AbstractArray{T, 3}, idx) where {T} = x[:, :, :, idx] .= d
_set_data_row!(x::AbstractArray{T, 5}, d::AbstractArray{T, 4}, idx) where {T} = x[:, :, :, :, idx] .= d

_set_data_row!(x::AbstractArray{T,1}, d::Base.ReshapedArray{T,1,M}, idx) where {T,M} = x[idx] = d[1]

function _set_data_row!(x::AbstractArray{T,2}, d::Base.ReshapedArray{T,1,M}, idx) where {T,M}
    x[:, idx] = d
end

function _set_data_row!(x::AbstractArray{T,2}, d::Base.ReshapedArray{T,2,M}, idx) where {T,M}
    x[:, idx] = reshape(d, :)
end

function _set_data_row!(x::AbstractArray{T,3}, d::Base.ReshapedArray{T,2,M}, idx) where {T,M}
    x[:, :, idx] .= d
end

function _set_data_row!(x::AbstractArray{T,4}, d::Base.ReshapedArray{T,3,M}, idx) where {T,M}
    x[:, :, :, idx] .= d
end

function _set_data_row!(x::AbstractArray{T,5}, d::Base.ReshapedArray{T,4,M}, idx) where {T,M}
    x[:, :, :, :, idx] .= d
end

####################################################################
# CUDA arrays
####################################################################
# Adding CuArray when buffer is on the CPU
function _set_data_row!(x::Array{T,N}, d::CuArray{T}, idx) where {T,N}
    _set_data_row!(x, d |> cpu_device(), idx)
end

# Adding Array when buffer is on the GPU
function _set_data_row!(x::CuArray{T,N}, d::Array{T}, idx) where {T,N}
    _set_data_row!(x, d |> gpu_device(), idx)
end
####################################################################


"""
    push!(buffer, data)

Adds data to the buffer, where data is an array of collections of types defined in
CircularBuffer._data_types returns row of data of added d
"""
function Base.push!(buffer::CB, data) where {CB<:CircularBuffer}
    ret = buffer._current_row

    for (idx, dat) in enumerate(data)
        _set_data_row!(buffer._stg_tuple[idx], data[idx], buffer._current_row)
    end

    buffer._current_row += 1
    if buffer._current_row > buffer._capacity
        buffer._current_row = 1
        buffer._full = true
    end

    return ret
end

function Base.push!(buffer::CB, data::NamedTuple) where {CB<:CircularBuffer}
    ret = buffer._current_row
    for k ∈ keys(buffer._stg_tuple)
        _set_data_row!(buffer._stg_tuple[k], data[k], buffer._current_row)
    end

    buffer._current_row += 1
    if buffer._current_row > buffer._capacity
        buffer._current_row = 1
        buffer._full = true
    end

    return ret
end

"""
    length(buffer)

Return the current amount of data in the circular buffer. If the full flag is true then we
return the size of the whole data frame.
"""
function Base.length(buffer::CircularBuffer)
    if buffer._full
        buffer._capacity
    else
        buffer._current_row-1::Int
    end
end

"""
    capacity(buffer)

Return the max number of elements the buffer can store.
"""
capacity(buffer::CircularBuffer) = buffer._capacity

function Base.getindex(buffer::CircularBuffer{TPL, TPS, NAMES}, idx) where {TPL, TPS, NAMES}
    (;zip(NAMES, (_get_data_row(buffer._stg_tuple[i], idx) for i in 1:length(NAMES)))...)
end

function Base.getindex(buffer::CircularBuffer, idx::Symbol)
    buffer._stg_tuple[idx]
end

Base.view(buffer::CircularBuffer{TPL, TPS, NAMES}, idx) where {TPL, TPS, NAMES} =
    NamedTuple{NAMES}(_get_data_row_view(buffer._stg_tuple[i], idx) for i in 1:length(NAMES))
