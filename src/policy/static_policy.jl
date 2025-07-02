"""
    StaticUniformPolicy{F<:AbstractFloat,E<:AbstractEnvironment} <: AbstractStaticPolicy
"""
struct StaticUniformPolicy{F<:AbstractFloat,E<:AbstractEnvironment} <: AbstractStaticPolicy
    _env::E

    function StaticUniformPolicy{F}(env::E) where {F<:AbstractFloat,E<:AbstractEnvironment}
        return new{F,E}(env)
    end
end

function StaticUniformPolicy(env::E) where {E}
    F = eltype(rand(action_space(env)))
    StaticUniformPolicy{F}(env)
end

continuous(p::StaticUniformPolicy) = continuous(action_space(p._env))
discrete(p::StaticUniformPolicy) = discrete(action_space(p._env))

function sample(u::StaticUniformPolicy, rng::AbstractRNG, f, states; num_samples = 1)
    sample(u::StaticUniformPolicy, rng, states; num_samples = 1)
end

function sample(u::StaticUniformPolicy, rng::AbstractRNG, states; num_samples = 1)
    if ndims(states) > ndims(observation_space(u._env))
        batch_size = size(states)[end]
    else
        batch_size = 1
    end

    as = action_space(u._env)
    samples = rand(rng, as, num_samples, batch_size)

    if batch_size == 1
        samples = reshape(samples, size(as)..., num_samples)
    end

    return samples
end

function logprob(u::StaticUniformPolicy{F}, f, state, action)::Vector{F} where {F}
    logprob(u::StaticUniformPolicy{F}, state, action)::Vector{F} where {F}
end

function logprob(u::StaticUniformPolicy{F}, state, action)::Vector{F} where {F}
    if ndims(action) == 2
        batch_size = size(action)[end]
    elseif ndims(action) == 1
        batch_size = 1
    else
        error("actions should be vectors but got input with $(ndims(action)) dimensions")
    end

    length = upper(action_space(env)) .- lower(action_space(env))
    pdf = oneunit(F) ./ length
    logpdf = sum(log.(pdf))

    return [logpdf for _ in 1:batch_size]
end
