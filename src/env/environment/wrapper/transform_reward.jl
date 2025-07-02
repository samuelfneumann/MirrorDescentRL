"""
    TransformReward{P} <: AbstractEnvironmentRewardWrapper where {P<:AbstractFloat}

Transform the reward via an arbitrary function.
"""
mutable struct TransformReward{P} <: AbstractEnvironmentRewardWrapper where {P<:AbstractFloat}
    _env::AbstractEnvironment
    _f
    _last_reward::AbstractFloat

	function TransformReward{P}(env::AbstractEnvironment, f) where {P<:AbstractFloat}
        if ! RLEnv.iscallable(f)
            error("f must be callable")
        end

        return new{P}(env, f, f(reward(env)))
    end
end

"""
	TransformReward{P}(env::AbstractEnvironment, f)
	TransformReward(env::AbstractEnvironment, f)

Constructor.

The type parameter `P` determines the type of the reward that is returned. If it is left
unspecified, then the type reward returned will be the same as that from the environment. If
`P` is specified, then the type of the returned reward will be `P`.

# Arguments
# -`env::AbstractEnvironment`: The environment to wrap
# -`f`: The function that returns the altered reward
"""
function TransformReward(env::AbstractEnvironment, f)
	P = typeof(reward(env))

    return TransformReward{P}(env, f)
end

function reward(t::TransformReward{P})::P where {P<:AbstractFloat}
    return t._last_reward
end

function reward(t::TransformReward{P}, r::AbstractFloat) where {P<:AbstractFloat}
    return convert(P, t._f(r))
end

function set_last_reward!(t::TransformReward, r::AbstractFloat)
    t._last_reward = r
end

function wrapped(t::TransformReward)::AbstractEnvironment
	return t._env
end
