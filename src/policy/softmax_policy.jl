# IMPORTANT: we never include the action shape dimension, as it is always 1!
import Lux: softmax

# SoftmaxPolicy distribution over N values using floating point
# precision F and integers I
# TODO: I'm not sure constraining inputs to type F is a good idea anymore
struct SoftmaxPolicy{F,N,I} <: AbstractDiscreteParameterisedPolicy
    function SoftmaxPolicy{F,N,I}() where {F,N,I<:Integer}
        return new{F,N,I}()
    end
end

# TODO: implement entropy and kl divergence functions

ExtendedDistributions.analytical_kl(::SoftmaxPolicy) = true
ExtendedDistributions.analytical_entropy(::SoftmaxPolicy) = true

function valid_fa(b::SoftmaxPolicy, env, fa)
    # Check to make sure that the approximator outputs the correct number of values
    out = fa(rand(observation_space(env)))
    if size(out)[1] != N
        error("expected approximator to output $N values for environment $env " *
            "but got $(size(out)[1])")
    end
end

# SoftmaxPolicy is always over discrete variables, so we never include the action shape when
# sampling etc., since it is always 1.
function SoftmaxPolicy{F}(env::AbstractEnvironment) where {F}
    as = action_space(env)

    if !(as isa ActorCritic.Discrete)
        error(
            "expected action space to be of type Spaces.Discrete but got $(typeof(as))"
        )
    elseif ndims(as) > 1
        error("expected a single-dimensional Discrete action space but got $(ndims(as))")
    end

    N = as.n[1]

    SoftmaxPolicy{F,N,Int}()
end

function SoftmaxPolicy(env::AbstractEnvironment)
    F = eltype(rand(observation_space(env)))
    SoftmaxPolicy{F}(env)
end

continuous(::SoftmaxPolicy) = false
discrete(::SoftmaxPolicy) = true
Base.eltype(::SoftmaxPolicy{T,N,I}) where {T,N,I} = Vector{I}

function _get_logits(
    s::SoftmaxPolicy{T,N}, model, model_θ, model_st, states,
) where {T,N}
    logits = model(states, model_θ, model_st)
end

function sample(
    s::SoftmaxPolicy{T,N,I},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray{T};
    num_samples=1,
) where {T,N,I}
    logits, model_st = _get_logits(s, model, model_θ, model_st, states)

    samples = if ndims(logits) > 1
        batch_size = size(logits)[end]
        s = [
                rand(rng, Categorical(softmax(logits[:, i])), num_samples)
                for i in 1:batch_size
        ]
        reduce(hcat, s)
    else
        rand(rng, Categorical(softmax(logits)), num_samples)
    end
    return samples, model_st
end

function prob(s::SoftmaxPolicy, model::Tabular, model_θ, model_st)
    logits, model_st = model(model_θ, model_st)
    return softmax(logits), model_st
end

function prob(
    s::SoftmaxPolicy{T},
    model,
    model_θ,
    model_st,
    states::AbstractArray{T},
) where {T}
    logits, model_st = _get_logits(s, model, model_θ, model_st, states)
    return softmax(logits), model_st
end

function logprob(s::SoftmaxPolicy, model::Tabular, model_θ, model_st)
    logits, model_st = model(model_θ, model_st)
    return logsoftmax(logits), model_st
end

function logprob(
    s::SoftmaxPolicy{T},
    model,
    model_θ,
    model_st,
    states::AbstractArray{T},
) where {T}
    logits, model_st = _get_logits(s, model, model_θ, model_st, states)
    return logsoftmax(logits), model_st
end

function logprob(
    s::SoftmaxPolicy{T},
    model,
    model_θ,
    model_st,
    states::AbstractArray{T},
    actions::Union{Int,AbstractArray{Int}},
) where {T}
    logprobs, model_st = logprob(s, model, model_θ, model_st, states)
    return _index_logprob(logprobs, actions), model_st
end

function _index_logprob(
    logprobs::AbstractArray{T, 2},
    actions::AbstractArray{Int, 2}, # batch of actions
)::AbstractArray{T} where{T}
    batch_size = size(logprobs, 2)
    return logprobs[[CartesianIndex(actions[j, i], i) for j in 1:2, i in 1:batch_size]]
end

function _index_logprob(
    logprobs::AbstractArray{T, 2},
    actions::AbstractArray{Int, 1}, # batch of actions
)::AbstractArray{T} where{T}
    batch_size = size(logprobs, 2)
    return logprobs[[CartesianIndex(actions[i], i) for i in 1:batch_size]]
end

function _index_logprob(
    logprobs::AbstractVector{T},
    action::Int,
)::T where{T}
    return logprobs[action]
end

function _index_logprob(
    logprobs::AbstractVector{T},
    actions::AbstractVector{Int},
)::Vector{T} where{T}
    return logprobs[actions]
end

function Distributions.mode(
    p::SoftmaxPolicy{T},
    model,
    model_θ,
    model_st,
    states::AbstractArray{T};
    num_samples = 1,
) where {T}
    all_probs = logprob(p, model, model_θ, model_state, states)

    return if ndims(all_probs) > 1
        batch_size = size(all_probs)[end]
        mode = argmax(all_probs; dims=1)
        mode = _extract_pos.(mode; dims=1)
        repeat(mode; inner = (1, num_samples))
    else
        mode = argmax(all_probs)
        [mode for _ in 1:num_samples]
    end
end

# kl divergence when probabilities are given
function Distributions.kldivergence(dist::SoftmaxPolicy, p, q; logprob::Bool)
    return if !logprob
        log_p = log.(p)
        log_q = log.(q)
        sum(p .* (log_p .- log_q); dims=1)
    else
        sum(exp.(p) .* (p .- q); dims=1)
    end
end

# Calculates KL(p || q)
# TODO: combine this with the Tabular implementation below
function Distributions.kldivergence(
    dist::SoftmaxPolicy,
    p_model,
    p_θ,
    p_st,
    q_model,
    q_θ,
    q_st,
    states,
)
    log_p_prob, p_st = logprob(dist, p_model, p_θ, p_st, states)
    log_q_prob, q_st = logprob(dist, q_model, q_θ, q_st, states)
    p_prob = exp.(log_p_prob)

    @tullio kl[i] := p_prob[j, i] * (log_p_prob[j, i] - log_q_prob[j, i])

    return kl, p_st, q_st
end

function Distributions.kldivergence(
    dist::SoftmaxPolicy,
    p_model,
    p_θ,
    p_st,
    q_θ,
    q_st,
    states,
)
    return kldivergence(dist, p_model, p_θ, p_st, p_model, q_θ, q_st, states)
end

function Distributions.kldivergence(
    dist::SoftmaxPolicy,
    p_model::Tabular,
    p_θ,
    p_st,
    q_model::Tabular,
    q_θ,
    q_st,
    state::Int,
)
    log_p_prob, p_st = logprob(dist, p_model, p_θ, p_st, state)
    log_q_prob, q_st = logprob(dist, q_model, q_θ, q_st, state)
    p_prob = exp.(log_p_prob)

    @tullio kl[i] := p_prob[j, i] * (log_p_prob[j, i] - log_q_prob[j, i])
    return kl, p_st, q_st
end

function Distributions.kldivergence(
    dist::SoftmaxPolicy,
    p_model::Tabular,
    p_θ,
    p_st,
    q_model::Tabular,
    q_θ,
    q_st,
    state::Vector{Int},
)
    log_p_prob, p_st = logprob(dist, p_model, p_θ, p_st, state)
    log_q_prob, q_st = logprob(dist, q_model, q_θ, q_st, state)
    p_prob = exp.(log_p_prob)

    @tullio kl[i] := p_prob[j, i] * (log_p_prob[j, i] - log_q_prob[j, i])

    return kl, p_st, q_st
end

function Distributions.kldivergence(
    dist::SoftmaxPolicy,
    p_model::Tabular,
    p_θ,
    p_st,
    q_model::Tabular,
    q_θ,
    q_st,
    state::Matrix{Int},
)
    @assert size(state, 1) == 1
    return kldivergence(dist, p_model, p_θ, p_st, q_model, q_θ, q_st, state[1, :])
end

function _extract_pos(ind::CartesianIndex; dims)
    return ind.I[dims]
end

# KL divergence when policies share same function approximator architecture
function Distributions.kldivergence(
    dist::SoftmaxPolicy,
    model::Tabular,
    p_θ,
    p_st,
    q_θ,
    q_st,
    states,
)
    return kldivergence(dist, model, p_θ, p_st, model, q_θ, q_st, states)
end

function Distributions.entropy(
    dist::SoftmaxPolicy,
    model,
    model_θ,
    model_st,
    states,
)
    lnπ, model_st = logprob(dist, model, model_θ, model_st, states)
    @tullio h[i] := -exp(lnπ[j, i]) * lnπ[j, i]
    return h, model_st
end

####################################################################
# Gradients of tabular-softmax policy functions
####################################################################
function _∇ln_softmax_tabular(θ_, s_t, a_t)
    θ = softmax(θ_[:, s_t])
    gs = zeros(Float32, first(size(θ)))
    ind = ones(Bool, first(size(θ)))
    ind[a_t] = 0f0
    gs[ind] .= -θ[ind]
    gs[a_t] = 1f0 - θ[a_t]
    return gs
end

function _∇ln_softmax_tabular(θ_, s_t; sum_over_actions::Bool)
    θ = softmax(θ_[:, s_t])
    return if !sum_over_actions
        @tullio gs[i, j] := ifelse(i == j, θ[i] * (1 - θ[i]), -θ[i] * θ[j])
    else
        @tullio gs[i] := ifelse(i == j, 1 - θ[i], -θ[j])
    end
end

function _∇_softmax_tabular(θ_, s_t, a_t)
    θ = softmax(θ_[:, s_t])
    gs = zeros(Float32, first(size(θ)))
    ind = ones(Bool, first(size(θ)))
    ind[a_t] = 0f0
    gs[ind] .= -θ[ind] * θ[a_t]
    gs[a_t] = θ[a_t] * (1f0 - θ[a_t])
    return gs
end

function _∇_softmax_tabular(θ_, s_t; sum_over_actions::Bool)
    θ = softmax(θ_[:, s_t])
    return if sum_over_actions
        @tullio gs[i] := ifelse(i == j, θ[i] * (1 - θ[i]), -θ[i] * θ[j])
    else
        @tullio gs[i, j] := ifelse(i ==j, 1 - θ[i], -θ[j])
    end
end

function _∇entropy_softmax_tabular(θ_, s_t)
    logθ = logsoftmax(θ_[:, s_t])
    θ = softmax(θ_[:, s_t])
    @tullio g_t_1[i] := θ[i] * (1f0 - θ[i]) * logθ[i]
    @tullio g_t_2[i] := ifelse(j != i, -θ[i] * θ[j] * logθ[j], 0f0)
    return -(g_t_1 .+ g_t_2)
end

function _∇kl_softmax_tabular(θ_, φ_, s_t)
    # if θ_ == φ_
    #     return spzeros(Float32, size(θ_, 1))
    # end
    θ = softmax(θ_[:, s_t])
    logθ = logsoftmax(θ_[:, s_t])
    φ = softmax(φ_[:, s_t])
    logφ = logsoftmax(φ_[:, s_t])

    @tullio g_t_1[i] := ifelse(j != i, -θ[i] * θ[j] * (logθ[j] - logφ[j]), 0f0)
    @tullio g_t_2[i] := θ[i] * (1f0 - θ[i]) * (logθ[i] - logφ[i])

    return g_t_1 .+ g_t_2
end
