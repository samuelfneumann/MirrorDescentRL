# This file implements rudimentary checkpoint handling
using Dates
using JLD2

struct CheckpointMetaData
    dir::String
    file::String

    function CheckpointMetaData(dir::String, file::String)
        return new(dir, file)
    end
end

checkpointfile(c::CheckpointMetaData) = joinpath(c.dir, c.file)
checkpointed(c::CheckpointMetaData) = isfile(checkpointfile(c))

function load_checkpoint(c::CheckpointMetaData)
    kwargs = nothing
    timestamp = Dates.now()
    if checkpointed(c)
        try
            JLD2.@load checkpointfile(c) kwargs timestamp
            @info "Loading checkpoint from ($timestamp): \n\tFile: $(checkpointfile(c))"
        catch e
            @info (
                "Could not load checkpoint from ($timestamp):\n" *
                "\tFile:\t$(checkpointfile(c))\n" *
                "\tError:\t$e\n" *
                "\tInfo:\tRestarting experiment, logging error message to stderr"
            )
            @error "ERROR: " exception=(e, catch_backtrace())
            rm(checkpointfile(c))
            kwargs = nothing
        end
    end
    return Checkpoint(c, kwargs, timestamp)
end

function load_checkpoint(dir, file)
    md = CheckpointMetaData(dir, file)
    return load_checkpoint(md)
end

Base.mkpath(c::CheckpointMetaData) = Base.mkpath(c.dir)


mutable struct Checkpointer
    md::CheckpointMetaData
    every::Int # Steps
    step_last_checkpointed::Int

    function Checkpointer(every::Int, md)
        step_last_checkpointed = 0
        if checkpointed(md)
            JLD2.@load checkpointfile(md) step_last_checkpointed
        end
        return new(md, every, step_last_checkpointed)
    end
end

function Checkpointer(every::Int, dir, file)
    md = CheckpointMetaData(dir, file)
    return Checkpointer(every, md)
end

checkpointfile(c::Checkpointer) = checkpointfile(c.md)
checkpointed(c::Checkpointer) = checkpointed(c.md)

function write_checkpoint(c::Checkpointer, step; kwargs...)
    if c.every <= 0
    elseif (step - c.step_last_checkpointed) >= c.every
        timestamp = Dates.now()
        @info "Saving checkpoint at step ($timestamp): $step"

        c.step_last_checkpointed = step
        step_last_checkpointed = step

        path = checkpointfile(c)
        JLD2.@save path kwargs timestamp step_last_checkpointed
    end
    return nothing
end

Base.mkpath(c::Checkpointer) = Base.mkpath(c.md)

struct Checkpoint{T}
    md::CheckpointMetaData
    data::T
    timestamp::DateTime
end

get_data(c::Checkpoint) = c.data
get_data(c::Checkpoint, key) = c.data[key]
checkpointfile(c::Checkpoint) = checkpointfile(c.md)
checkpointed(c::Checkpoint) = checkpointed(c.md)
Base.mkpath(c::Checkpoint) = Base.mkpath(c.md)
metadata(c::Checkpoint) = c.md
