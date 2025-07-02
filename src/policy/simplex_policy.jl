"""
    SimplexPolicy <: AbstractDiscreteParameterisedPolicy

Represents a simplex policy. The policy is parameterized by a matrix of
probabilities of size (|S|, |A|), where |S| is the number of states and |A| is the number of
actions.

This implementation is over-parameterized since we learn N action probabilities for N
actions. In reality, we only need to learn (N-1) prbabilities for N actions, since we must
always satisfy the sum-to-one constraint.
"""
struct SimplexPolicy <: AbstractDiscreteParameterisedPolicy end

SimplexPolicy(::AbstractEnvironment) = SimplexPolicy()

function set_action_probs(
    p::SimplexPolicy,
    probs::AbstractVector,
    model::Tabular,
    θ,
    st,
    state::Int,
)
    _check_probability_vector(probs)
    ps = setcol(model, state, probs, θ)
    return ps
end

function sample(
    p::SimplexPolicy,
    rng::AbstractRNG,
    model::Tabular,
    θ,
    st,
    states::AbstractArray;
    num_samples = 1,
)
    probs, st = prob(p, model, θ, st, states)

    # TODO: maybe we can remove this and just work with single samples or batches with batch
    # size > 1, like with IIDPolicy (there it is vector/matrix, here it is
    # int/vector{int})... When we have a batch size of 1. The issue is that, similar to the
    # SoftmaxPolicy implementation, we want to be consistent, hence outputs are always
    # vectors, even though we only really need int outputs. This makes things more
    # consistent between continuous-action and discrete-action policies.
    # if ndims(probs) == 2 && size(probs, 2) == 1
    #     # Unsqeeze batch dimension
    #     probs = probs[:, 1]
    # end

    samples = if ndims(probs) == 1
        [rand(rng, Categorical(probs)) for _ in 1:num_samples]
    elseif ndims(probs) == 2 # batch
        weights = [collect(ProbabilityWeights(prob)) for prob in eachcol(probs)]
        out = [
            [rand(rng, Categorical(w)) for _ in 1:num_samples] for w in weights
        ]
        out = stack(out)
        out
    end

    return samples, st
end

function prob(p::SimplexPolicy, model::Tabular, θ, st)
    probs, st = model(θ, st)
    _check_probability_vector(probs)
    return probs, st
end

function prob(p::SimplexPolicy, model::Tabular, θ, st, states)
    probs, st = model(states, θ, st)
    _check_probability_vector(probs)
    return probs, st
end

function prob(p::SimplexPolicy, model::Tabular, θ, st, states, actions)
    probs, st = prob(p, model, θ, st)
    _check_probability_vector(probs)
    return probs[[CartesianIndex(actions[i], i) for i in 1:length(states)]], st
end

function logprob(p::SimplexPolicy, model::Tabular, θ, st)
    probs, st = prob(p, model, θ, st)
    return log.(probs), st
end

function logprob(p::SimplexPolicy, model::Tabular, θ, st, states)
    probs, st = prob(p, model, θ, st, states)
    return log.(probs), st
end

function logprob(
    p::SimplexPolicy,
    model::Tabular,
    θ,
    st,
    states,
    actions::Union{Int,AbstractArray{Int}},
)
    lp, st = logprob(p, model, θ, st, states)
    return _index_logprob(lp, actions), st
end

# Calculates KL(p || q)
function Distributions.kldivergence(
    dist::SimplexPolicy,
    p_model::Tabular,
    p_θ,
    p_st,
    q_model::Tabular,
    q_θ,
    q_st,
    state::Int,
)
    p_prob, p_st = prob(dist, p_model, p_θ, p_st, state)
    q_prob, q_st = prob(dist, q_model, q_θ, q_st, state)
    # kl = sum(p_prob .* (log.(p_prob) - log.(q_prob)))
    @tullio kl[i] := p_prob[j, i] * (log(p_prob[j, i]) - log(q_prob[j, i]))
    return kl, p_st, q_st
end

function Distributions.kldivergence(
    dist::SimplexPolicy,
    p_model::Tabular,
    p_θ,
    p_st,
    q_model::Tabular,
    q_θ,
    q_st,
    state::Vector{Int},
)
    p_prob, p_st = prob(dist, p_model, p_θ, p_st, state)
    q_prob, q_st = prob(dist, q_model, q_θ, q_st, state)

    @tullio kl[i] := p_prob[j, i] * (log.(p_prob)[j, i]  - log(q_prob[j, i]))
    # kl = sum(p_prob .* (log.(p_prob) - log.(q_prob)); dims=1)[1, :]

    return kl, p_st, q_st
end

function Distributions.kldivergence(
    dist::SimplexPolicy,
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

# KL divergence when policies share same function approximator architecture
function Distributions.kldivergence(
    dist::SimplexPolicy,
    model::Tabular,
    p_θ,
    p_st,
    q_θ,
    q_st,
    states,
)
    return kldivergence(dist, model, q_θ, p_st, model, q_θ, q_st, states)
end

function Distributions.mode(
    p::SimplexPolicy,
    model::Tabular,
    θ,
    st,
    states;
    num_samples = 1,
)
    probs, st = model(states, θ, st)
    _check_probability_vector(probs)

    mode = if ndims(probs) == 1
        mode = argmax(probs)
        [mode for _ in 1:num_samples]
    elseif ndims(probs) == 2
        batch_size = size(probs)[end]
        mode = argmax(probs; dims=1)
        mode = _extract_pos.(mode; dims=1)
        repeat(mode; inner = (1, num_samples))
    end

    return mode, st
end

function Distributions.entropy(
    dist::SimplexPolicy,
    model,
    model_θ,
    model_st,
    states,
)
    probs, model_st = prob(dist, model, model_θ, model_st, states)
    lnπ, model_st = logprob(dist, model, model_θ, model_st, states)

    @tullio h[i] := ifelse(
        probs[j, i] == zero(probs[j, i]),
        zero(probs[j, i]),
        -probs[j, i] * lnπ[j, i],
    )

    return h, model_st
end

function _check_probability_vector(probs::AbstractVector)
    @assert sum(probs) ≈ 1f0 "expected a probability vector but got $probs"
end

function _check_probability_vector(probs::AbstractMatrix)
    for prob in eachcol(probs)
        _check_probability_vector(prob)
    end
end

function _∇kl_simplex_tabular(θ, φ, s_t)
    # return log.(θ[:, s_t]) .+ 1 .- log.(φ[:, s_t])
    return log.(θ[:, s_t]) .- log.(φ[:, s_t])
end

function _∇entropy_simplex_tabular(θ, s_t)
    # Entropy of a simplex vector p is ∑ₐln(pₐ) + 1
    #
    # But, we are not simply using a simplex vector, we are using a parameterized policy
    # π_θ, which happens to be represented by a simplex vector. Hence, the gradient is:
    #
    #   ∑ₐln(π_θ(a∣s)) ∇_θ π_θ + 𝔼_{π_θ}[∇_θ ln(π_θ(a∣s))] = ∑ₐln(π_θ(a∣s)) ∇_θ π_θ + 0
    #
    #   since the expected gradient of the log probability of a distribution is 0.
    return log.(θ[:, s_t]) # .+ 1
end

function _∇ln_simplex_tabular(θ, s_t, a_t)
    grad = spzeros(size(θ, 1))
    grad[a_t] = inv(θ[a_t, s_t])
    return grad
end

function _∇ln_simplex_tabular(θ, s_t; sum_over_actions::Bool)
    return @tullio gs[i, i] =  inv(θ[:, s_t][i])
end

function _∇_simplex_tabular(θ, s_t, a_t)
    grad = zeros(size(θ, 1))
    grad[a_t] = 1
    return grad
end

function _∇_simplex_tabular(θ, s_t; sum_over_actions::Bool)
    return if sum_over_actions
        ones(size(θ, 1))
    else
        Diagonal(ones(size(θ, 1)))
    end
end
