mutable struct RewardNoise{
    E,
    D<:ContinuousUnivariateDistribution,
    R<:AbstractRNG,
} <: AbstractEnvironmentRewardWrapper
    const _env::E
    const _p::Float32 # ∈ (0, 1]
    const _dist::D
    const _rng::R
    _last_reward::Float32

    function RewardNoise(env::E, p::Real, dist::D, rng::R) where {E,D,R}
        @assert 0 < p <= 1
        return new{E,D,R}(env, p, dist, rng, 0f0)
    end
end

function reward(r::RewardNoise, rew::Real)::Float32
    return if r._p != 1
        ε = rand(r._rng, Float32)
        if ε < r._p
            rew + rand(r._rng, r._dist)
        else
            rew
        end
    else
        rew + rand(r._rng, r._dist)
    end
end

reward(r::RewardNoise) = r._last_reward
wrapped(r::RewardNoise) = r._env
set_last_reward!(r::RewardNoise, rew::Real) = r._last_reward = rew
