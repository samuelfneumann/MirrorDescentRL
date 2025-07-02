"""
    UpdateRatio

UpdateRatio stores a ratio of number of updates to perform per environment step
"""
struct UpdateRatio
    _updates::UInt
    _steps::UInt
end

UpdateRatio(r::Rational) = UpdateRatio(numerator(r), denominator(r))
UpdateRatio(t::NTuple{2, Integer}) = UpdateRatio(t[1], t[2])
UpdateRatio(n::Integer) = UpdateRatio(n, 1)
UpdateRatio(f::AbstractFloat) = UpdateRatio(convert(Rational, f))
UpdateRatio(u::UpdateRatio) = u

function UpdateRatio(t::Vector{<:Integer})
    @assert length(t) == 2
    return UpdateRatio(t[1], t[2])
end

function UpdateRatio(
    t::NamedTuple{(:updates, :steps), Tuple{T1, T2}},
) where {T1<:Integer,T2<:Integer}
    return UpdateRatio(t.updates, t.steps)
end

function UpdateRatio(
    t::NamedTuple{(:steps, :updates), Tuple{T1, T2}},
) where {T1<:Integer,T2<:Integer}
    return UpdateRatio(t.updates, t.steps)
end

"""
    should_update(u::UpdateRatio, current_step::Int, current_update::Int)::Bool

Returns whether or not you should update, based on the updating schedule of the
`_UpdateRatio`.
"""
function should_update(u::UpdateRatio, current_step::Integer, current_update::Integer)::Bool
    @assert current_update > 0
    @assert current_step > 0
    return mod(current_step, u._steps) == 0 && current_update <= u._updates
end

updates_per_step(u::UpdateRatio) = (u._updates, u._steps)

