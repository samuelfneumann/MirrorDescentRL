"""
    CastObservation{E,T} <: AbstractEnvironmentObservationWrapper

Cast the observations of environment with type `E` to type `AbstractArray{T}`
"""
struct CastObservation{E,T} <: AbstractEnvironmentObservationWrapper
    _env::E

    function CastObservation{T}(env::E) where {E<:AbstractEnvironment,T}
        return new{E,T}(env)
    end
end

function observation(c::CastObservation{E,T}, obs::AbstractArray) where{E,T}
    if eltype(obs) <: AbstractFloat && T <: AbstractFloat
        return convert.(T, obs)
    elseif eltype(obs) <: AbstractFloat && T <: Unsigned
        return trunc.(T, abs.(obs))
    elseif eltype(obs) <: AbstractFloat && T <: Integer
        return trunc.(T, obs)
    end
end

function wrapped(c::CastObservation{E})::E where {E}
	return c._env
end
