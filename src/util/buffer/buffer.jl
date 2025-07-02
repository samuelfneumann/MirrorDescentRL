# TODO: it might be a good idea to make ERs immutable and pass around their state too.

export AbstractReplay, ExperienceReplay

import DSP: filt, PolynomialRatio
using Random
import Random

include("circular_buffer.jl")
include("sum_tree.jl")

# ####################################
# Abstract Replay Buffer
# ####################################
abstract type AbstractReplay end

proc_state(er::AbstractReplay, x) = identity(x)

Base.keys(er::AbstractReplay) = 1:length(er)

function Base.iterate(asr::AbstractReplay)
    state = 1
    result = asr[state]
    state += 1
    (result, state)
end

function Base.iterate(asr::AbstractReplay, state::Integer)
    if state > length(asr)
        return nothing
    end

    result = asr[state]
    state += 1

    (result, state)
end

# ####################################
# Vector Experience Replay Buffer
# ####################################
mutable struct VectorExperienceReplay{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
    sizes::Tuple{Int,Int,Int,Int,Int} # s_t, a_t, r_tp1, s_tp1, γ_tp1
end

Adapt.@adapt_structure ActorCritic.VectorExperienceReplay

function reset!(er::VectorExperienceReplay)
    reset!(er.buffer)
end

function VectorExperienceReplay(capacity, type, sizes, column_name)
    shape = (sum(sizes),)
    cb = CircularBuffer(capacity, (type,), shape, (column_name,))
    VectorExperienceReplay(cb, sizes)
end

function VectorExperienceReplay(env::AbstractEnvironment, capacity::Int; F=Float32)
    msg = "VectorExperienceReplay expects vector observations"
    @assert ndims(observation_space(env)) == 1 msg

    msg = "VectorExperienceReplay expects vector actions"
    @assert ndims(action_space(env)) == 1 msg

    obs_size = size(observation_space(env))[1]
    act_size = if continuous(action_space(env))
        size(action_space(env))
    else
        1
    end
    r_size = γ_size = 1

    sizes = (obs_size, act_size, r_size, obs_size, γ_size)

    return VectorExperienceReplay(capacity, F, sizes, :item)
end

Base.length(er::VectorExperienceReplay) = length(er.buffer)
Base.getindex(er::VectorExperienceReplay, idx) = er.buffer[idx]
Base.view(er::VectorExperienceReplay, idx) = @view er.buffer[idx]

function Base.push!(er::VectorExperienceReplay, experience::AbstractVector)
    push!(er.buffer, (experience,))
end

function Base.push!(er::VectorExperienceReplay, experience::AbstractVector...)
    push!(er.buffer, (reduce(vcat, experience),))
end

function Random.rand(er::VectorExperienceReplay, batch_size::Int)
    return Random.rand(Random.GLOBAL_RNG, er, batch_size)
end

function Random.rand(rng::Random.AbstractRNG, er::VectorExperienceReplay, batch_size::Int)
    # Note: for some reason, if the ER buffer is on the cpu and we use a CUDA.RNG to sample,
    # we get scalar indexing on the gpu, which is not allowed. I'm not sure why this
    # happens, but it is a case that we **shouldn't** ever run into.
    #
    # using any type of RNG works fine if the ER buffer is on the gpu, and if it is on the
    # cpu, then we will only ever be using a cpu-based RNG.
    #
    # Further, note that the VectorExperienceReplay will not exactly reproduce the results
    # of an ExperienceReplay with vector observations due to how integers are sampled in
    # this function, and the corresponding one for ExperienceReplay. In this implementation,
    # we sample random number in a way that avoids scalar indexing (on the next line)
    idx = mod.(rand(rng, Int32, batch_size), length(er)) .+ 1

    out = er[idx]

    start_ = cumsum((0, er.sizes...))[begin:end-1] .+ 1
    end_ = cumsum(er.sizes)

    s_t, a_t, r_tp1, s_tp1, γ_tp1  = collect(
        out.item[start_[i]:end_[i], :] for i in 1:length(start_)
    )

    return (s_t=s_t, a_t=a_t, r_tp1=r_tp1[1, :], s_tp1=s_tp1, γ_tp1=γ_tp1[1, :])
end

function Base.get(er::VectorExperienceReplay)
    idx = 1:length(er)
    out = er[idx]
    start_ = cumsum((0, er.sizes...))[begin:end-1] .+ 1
    end_ = cumsum(er.sizes)
    s_t, a_t, r_tp1, s_tp1, γ_tp1  = collect(
        out.item[start_[i]:end_[i], :] for i in 1:length(start_)
    )
    return (s_t=s_t, a_t=a_t, r_tp1=r_tp1[1, :], s_tp1=s_tp1, γ_tp1=γ_tp1[1, :])
end

# ####################################
# GAEBuffer
#
# A GAE Buffer stores trajectories and returns all stored trajectories
# See: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py
#
# We use the notations (γ, λ) defined in: https://arxiv.org/abs/1506.02438
# ####################################
mutable struct GAEBuffer{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
    path_start_index::Int
    γ::Float32
    λ::Float32

    # TODO: a more efficient implementation: only store path lengths and use those to get
    # G_0
    G_0::Vector{Float32}
    path_lengths::Vector{Int}

    function GAEBuffer(buffer::CB, γ, λ) where {CB}
        return new{CB}(buffer, 1, γ, λ, Float32[], Int[])
    end
end

λ(er::GAEBuffer) = er.λ
γ(er::GAEBuffer) = er.γ

# Adapt.@adapt_structure ActorCritic.GAEBuffer

function reset!(er::GAEBuffer)
    reset!(er.buffer)
    er.G_0 = Float32[]
    er.path_lengths = Int[]
    er.path_start_index = er.buffer._current_row
    return nothing
end

function GAEBuffer(capacity, γ, λ, types, shapes, column_names)
    cb = CircularBuffer(capacity, types, shapes, column_names)
    GAEBuffer(cb, γ, λ)
end

function GAEBuffer(env::AbstractEnvironment, capacity::Int, γ, λ; F=Float32)
    obs_size = size(observation_space(env))
    act_size = if continuous(action_space(env))
        size(action_space(env))
    else
        1
    end
    r_size = γ_size = 1

    obs_type = eltype(rand(observation_space(env)))
    act_type = eltype(rand(action_space(env)))
    if F === nothing
        F = typeof(γ(env))
    end

    types = (obs_type, act_type, F, F, F, F, F)
    shapes = (obs_size, act_size, 1, 1, 1, 1, 1)
    names = (:s_t, :a_t, :r_tp1, :γ_tp1, :v_t, :G_t, :A_t)

    return GAEBuffer(capacity, γ, λ, types, shapes, names)
end

function finish_path!(er::GAEBuffer, last_val)
    last_index = full(er.buffer) ? er.buffer._capacity : er.buffer._current_row - 1

    r_tp1 = er.buffer._stg_tuple.r_tp1[er.path_start_index:last_index]
    r_tp1 = [r_tp1..., last_val]

    state_values = er.buffer._stg_tuple.v_t[er.path_start_index:last_index]
    state_values = [state_values..., last_val]

    v_tp1 = state_values[begin+1: end]
    v_t = state_values[begin:end-1]
    δ = r_tp1[begin:end-1] .+ er.γ .* v_tp1 .- v_t

    f = PolynomialRatio([1f0], [1f0, -er.γ * er.λ])
    er.buffer._stg_tuple.A_t[er.path_start_index:last_index] .= reverse!(filt(
        f, reverse!(δ),
    ))

    f = PolynomialRatio([1f0], [1f0, -er.γ])
    er.buffer._stg_tuple.G_t[er.path_start_index:last_index] .= reverse!(filt(
        f,
        reverse!(r_tp1),
    ))[begin:end-1]

    G_0 = er.buffer._stg_tuple.G_t[er.path_start_index]
    push!(er.G_0, G_0)
    path_length = last_index - er.path_start_index + 1
    push!(er.path_lengths, path_length)

    er.path_start_index = er.buffer._current_row

    return nothing
end

Lazy.@forward GAEBuffer.buffer full
Base.firstindex(er::GAEBuffer) = 1
Base.lastindex(er::GAEBuffer) = er.path_start_index - 1
Base.length(er::GAEBuffer) = length(er.buffer)
Base.getindex(er::GAEBuffer, idx) = er.buffer[idx]
Base.view(er::GAEBuffer, idx) = @view er.buffer[idx]

# Push onto the buffer a tuple of (s_t, a_t, r_tp1, γ_tp1, v_t)
Base.push!(er::GAEBuffer, experience...) = push!(er, experience)
function Base.push!(er::GAEBuffer, experience)
    @assert !full(er.buffer) "buffer is full"
    @assert length(experience) == 5

    if !(experience[4] ≈ er.γ) && experience[4] != 0f0
        # Ensure received discount == discount used by buffer.
        #
        # Currently, we do not support proper advantage estimation in the finish_path!
        # function for γ != er.γ
        msg = (
            "buffer uses γ=$(er.γ), but got $(experience[4]): this may produce " *
            "unexpected results"
        )
        @warn msg
    end

    push!(er.buffer, experience)
end

function Base.get(er::GAEBuffer)
    cb = er.buffer
    last_index = full(er.buffer) ? er.buffer._capacity : er.buffer._current_row - 1
    data = er[begin:last_index]

    return (
        s_t = data.s_t,
        a_t = data.a_t,
        path_lengths = er.path_lengths,
        G_0 = er.G_0,
        G_t = data.G_t,
        A_t = data.A_t,
        γ_tp1 = data.γ_tp1,
    )
end

# ####################################
# Experience Replay Buffer
# ####################################
mutable struct ExperienceReplay{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
end

Adapt.@adapt_structure ActorCritic.ExperienceReplay

function reset!(er::ExperienceReplay)
    reset!(er.buffer)
end

function ExperienceReplay(capacity, types, shapes, column_names)
    cb = CircularBuffer(capacity, types, shapes, column_names)
    ExperienceReplay(cb)
end

function ExperienceReplay(env::AbstractEnvironment, capacity::Int; F=Float32)
    obs_size = size(observation_space(env))
    act_size = if continuous(action_space(env))
        size(action_space(env))
    else
        1
    end
    r_size = γ_size = 1

    obs_type = eltype(rand(observation_space(env)))
    act_type = eltype(rand(action_space(env)))
    if F === nothing
        F = typeof(γ(env))
    end

    types = (obs_type, act_type, F, obs_type, F)
    shapes = (obs_size, act_size, r_size, obs_size, γ_size)
    names = (:s_t, :a_t, :r_tp1, :s_tp1, :γ_tp1)

    return ExperienceReplay(
        capacity,
        types,
        shapes,
        names,
    )
end

Base.length(er::ExperienceReplay) = length(er.buffer)
Base.getindex(er::ExperienceReplay, idx) = er.buffer[idx]
Base.view(er::ExperienceReplay, idx) = @view er.buffer[idx]

Base.push!(er::ExperienceReplay, experience) = push!(er.buffer, experience)
Base.push!(er::ExperienceReplay, experience...) = push!(er.buffer, experience)

function Random.rand(er::ExperienceReplay, batch_size::Int)
    Random.rand(Random.GLOBAL_RNG, er, batch_size)
end

function Random.rand(rng::Random.AbstractRNG, er::ExperienceReplay, batch_size::Int)
    idx = rand(rng, 1:length(er), batch_size)
    return er[idx]
end

function Base.get(er::ExperienceReplay)
    idx = 1:length(er)
    return er[idx]
end

# ####################################
# Sequence Replay Buffer
# ####################################
abstract type AbstractSequenceReplay <: AbstractReplay end

mutable struct SequenceReplay{CB} <: AbstractSequenceReplay
    buffer::CB
    place::Int64
end

function SequenceReplay(size, types, shapes, column_names)
    cb = CircularBuffer(size, types, shapes, column_names)
    SequenceReplay(cb, 1)
end

function reset!(er::SequenceReplay)
    reset!(er.buffer)
    er.place = 1
end

Base.length(er::SequenceReplay) = length(er.buffer)
Base.getindex(er::SequenceReplay, idx) = er.buffer[idx]
Base.view(er::SequenceReplay, idx) = @view er.buffer[idx]
Base.push!(er::SequenceReplay, experience...) = push!(er, experience)

function Base.push!(er::SequenceReplay, experience)
    if er.buffer._full
        er.place = (er.place % capacity(er.buffer)) + 1
    end
    push!(er.buffer, experience)
end

function Random.rand(er::SequenceReplay, batch_size, seq_length)
    Random.rand(Random.GLOBAL_RNG, er, batch_size, seq_length)
end

function Random.rand(rng::Random.AbstractRNG, er::SequenceReplay, batch_size, seq_length)
    start_inx = rand(rng, 1:(length(er) + 1 - seq_length), batch_size)
    e = [view(er, start_inx .+ (i-1)) for i ∈ 1:seq_length]
    start_inx, e
end

mutable struct EpisodicSequenceReplay{CB} <: AbstractSequenceReplay
    buffer::CB
    place::Int64
    terminal_locs::Vector{Int}
    terminal_symbol::Symbol
end

function EpisodicSequenceReplay(size, types, shapes, column_names; terminal_symbol = :t)
    cb = CircularBuffer(size, types, shapes, column_names)
    EpisodicSequenceReplay(cb, 1, Int[], terminal_symbol)
end

function reset!(er::EpisodicSequenceReplay)
    reset!(buffer)
    er.place = 1
    er.terminal_locs = Int[]
end

Base.length(er::EpisodicSequenceReplay) = length(er.buffer)
Base.getindex(er::EpisodicSequenceReplay, idx) = er.buffer[idx]
Base.view(er::EpisodicSequenceReplay, idx) = @view er.buffer[idx]

function Base.push!(er::EpisodicSequenceReplay, experience)
    if er.buffer._full
        er.place = (er.place % capacity(er.buffer)) + 1
    end
    push!(er.buffer, experience)
end

function get_episode_ends(er::EpisodicSequenceReplay)
    # TODO: n-computations. Maybe store in a cache?
    findall((exp)->exp::Bool, er.buffer._stg_tuple[er.terminal_symbol])
end

function get_valid_starting_range(s, e, seq_length)
    if e - seq_length <= s
        s:s
    else
        (s:e-seq_length)
    end
end

function get_valid_indicies(er::EpisodicSequenceReplay, seq_length)
    # episode_ends = get_episode_ends(er)
    1:(length(er) + 1 - seq_length)
end

function get_sequence(er::EpisodicSequenceReplay, start_ind, max_seq_length)
    # ret = [view(er, start_ind)]
    ret = [er[start_ind]]
    er_size = length(er)
    if ((start_ind + 1 - 1) % er_size) + 1 == er.place ||
        ret[end][er.terminal_symbol][]::Bool
        return ret
    end

    for i ∈ 1:(max_seq_length-1)
        push!(ret, er[(((start_ind + i - 1) % er_size) + 1)])
        if ret[end][er.terminal_symbol][]::Bool || ((start_ind + i + 1 - 1) % er_size) + 1 == er.place
            break
        end
    end

    return ret
end

function Random.rand(er::EpisodicSequenceReplay, batch_size, max_seq_length)
   Random.rand(Random.GLOBAL_RNG, er, batch_size, max_seq_length)
end

function Random.rand(
    rng::Random.AbstractRNG, er::EpisodicSequenceReplay, batch_size, max_seq_length,
)
    # get valid starting indicies
    valid_inx = get_valid_indicies(er, 1)
    start_inx = rand(rng, valid_inx, batch_size)
    exp = [get_sequence(er, si, max_seq_length) for si ∈ start_inx]
    start_inx, exp
    # padding and batching handled by agent.
end

# ####################################
# Experience Replay Buffer for Images
# ####################################
include("image_replay.jl")
