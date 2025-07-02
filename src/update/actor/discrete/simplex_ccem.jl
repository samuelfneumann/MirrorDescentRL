const _simplex_proj_atol = 1f-7

struct SimplexCCEM{S} <: AbstractActorUpdate
    _temperature::Float32
    _solver::S

    # Trick to ensure the policy is always slightly stochastic. This may be needed to
    # improve numerical stability when calculating entropy in the policy performance
    # gradient and when using soft action values, which also use entropy. If the policy
    # becomes deterministic, then the entropy will be -Inf.
    _ensure_stochastic::Bool
    _minp::Float32

    function SimplexCCEM(τ, solver, ensure_stochastic, minp)
        return new{typeof(solver)}(τ, solver, ensure_stochastic, minp)
    end
end

function SimplexCCEM(τ; solver=SortSimplexProjection(), ensure_stochastic=true, minp=1f-7)
    return SimplexCCEM(τ, solver, ensure_stochastic, minp)
end

function setup(
    up::SimplexCCEM,
    ::AbstractEnvironment,
    π::SimplexPolicy,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG;
)
    return UpdateState(
        up,
        optim,
        (optim = Optimisers.setup(optim, π_θ), rng=Lux.replicate(rng)),
    )
end

function update(
    st::UpdateState{SimplexCCEM{S}},
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    states,             # Must be >= 2D
) where {S}
    up = st._update
    rng = st._state.rng
    optim_state = st._state.optim

    batch_size = size(states, 2)
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)

    # Find action of maximal value, breaking ties randomly
    a_maxs = mapslices(x -> argmax_break_ties_randomly(rng, x), q; dims=1)

    # Construct gradient
    gs = zero(eltype(π_θ.layer_1), π_f)
    treemap!(gs) do g_i
        probs, π_st = prob(π, π_f, π_θ, π_st)
        for i in 1:batch_size
            s_t = states[1, i]
            a_t = a_maxs[i]
            a_t_prob = probs[a_t, s_t]

            if up._temperature > 0f0
                Δ =  up._temperature .* log.(probs[:, s_t])
                Δ[a_t] -= inv(a_t_prob)
                g_i[:, s_t] .+= (Δ ./ batch_size)
            else
                g_i[a_t, s_t] -= (inv(a_t_prob) ./ batch_size)
            end
        end
        g_i
    end

    # Perform the mirror descent update, with an L2 norm mirror map
    optim_state, π_θ = Optimisers.update(optim_state, π_θ, gs)

    # Project back onto the simplex
    π_θ = treemap!(π_θ) do θ
        θ = simplex_project!(θ, up._solver)
        _ensure_stochastic!(π, π_f, θ; minp=up._minp)
    end

    return UpdateState(
        st._update,
        st._optim,
        (optim=optim_state, rng=rng),
    ), π_θ, π_st, qf_st
end

####################################################################
# Projecting onto the simplex using a Euclidean projection
####################################################################
# SortSimplexProjection implements a projection onto the simplex using a sorting method. It
# has worst-case O(NlogN) time complexity if using quicksort
#
#   https://angms.science/doc/CVX/Proj_simplex.pdf
#   https://link.springer.com/article/10.1007/BF01580223
#   https://optimization-online.org/wp-content/uploads/2014/08/4498.pdf
struct SortSimplexProjection end

positive(v::Real) = max(v, 0)

# Equality constraints: ∑xᵢ - 1 = 0 => ∑(yᵢ - μ)₊ - 1 = 0
h(y, μ) = sum(positive.(y .- μ)) - one(μ)
dh_dμ(y, μ) = -sum((y .- μ) .> 0)

function simplex_project(y::AbstractMatrix, args...; kwargs...)
    new_y = []
    for col in eachcol(y)
        push!(new_y, simplex_project(col, args...; kwargs...))
    end
    return hcat(new_y...)
end

function simplex_project!(y::AbstractMatrix, args...; kwargs...)
    for (i, col) in enumerate(eachcol(y))
        y[:, i] .= simplex_project(col, args...; kwargs...)
    end
    return y
end

function simplex_project(
    y::AbstractVector, solver::Roots.AbstractSecantMethod, init=0f0,
)
    # https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    if all(y .> 0) && isapprox(sum(y), 1; atol=_simplex_proj_atol)
        return y
    end

    obj = μ -> h(y, μ)

    μ = find_zero(obj, init, solver)
    return positive.(y .- μ)
end

function simplex_project(
    y::AbstractVector, solver::Roots.AbstractNewtonLikeMethod, init=0f0
)
    # https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    if all(y .> 0) && isapprox(sum(y), 1; atol=_simplex_proj_atol)
        return y
    end

    obj = μ -> h(y, μ)
    grad = μ -> dh_dμ(y, μ)

    μ = find_zero((obj, grad), init, solver)
    return positive.(y .- μ)
end

function simplex_project(
    y::AbstractVector, solver::SortSimplexProjection,
)
    if all(y .> 0) && isapprox(sum(y), 1; atol=_simplex_proj_atol)
        return y
    end
    # https://angms.science/doc/CVX/Proj_simplex.pdf
    # https://link.springer.com/article/10.1007/BF01580223
    # https://optimization-online.org/wp-content/uploads/2014/08/4498.pdf
    n_actions = length(y)
    u = sort(y; rev=true)
    K = nothing
    for i in eachindex(u)
        z = (sum(u[begin:i]) - 1) / i
        K = z < u[i] ? i : K
    end
    if K === nothing
        error("could not find K")
    end
    τ = (sum(u[begin:K]) - 1) / K
    out = positive.(y .- τ)
    return out
end

function simplex_project(
    y::AbstractVector, solver::Roots.AbstractBracketingMethod, bracket=(-100f0, 100f0)
)
    if all(y .> 0) && isapprox(sum(y), 1; atol=_simplex_proj_atol)
        return y
    end

    obj = μ -> h(y, μ)
    grad = μ -> dh_dμ(y, μ)

    μ = find_zero(obj, bracket, solver)
    return positive.(y .- μ)
end

function simplex_project(::Tabular, ps, args...)
    return (layer_1 = simplex_project(ps.layer_1, args...),)
end
####################################################################

"""
    _ensure_stochastic!
    _ensure_stochastic

Ensures a probability vector is non-zero coordinate-wise. When the input is a softmax
vector of logits, ensures that all logits are such that all coordinates have non-zero
probability after application of the softmax function.
"""
function _ensure_stochastic! end
function _ensure_stochastic end

function _ensure_stochastic!(π, π_f::Tabular, π_θ::AbstractMatrix; minp::F) where {F}
    for state in 1:size(π_θ, 2)
        π_θ = _ensure_stochastic!(π, π_f, π_θ, state; minp)
    end
    return π_θ
end

function _ensure_stochastic!(
    ::SimplexPolicy, π_f::Tabular, π_θ::AbstractMatrix, state; minp::F,
) where {F}
    # Actions with non-zero probability
    zero_prob_actions = π_θ[:, state] .< minp
    if !any(zero_prob_actions)
        return π_θ
    end
    nonzero_prob_actions = (!).(zero_prob_actions)

    # Calculate amount to decrease probability of actions denoted by a above
    n_zero_prob_actions = sum(zero_prob_actions)
    n_nonzero_prob_actions = sum(nonzero_prob_actions)
    scale = n_zero_prob_actions / n_nonzero_prob_actions

    # Increase probability of 0-probability actions
    π_θ[zero_prob_actions, state] .+= minp

    # Decrease probability of non-zero probability actions
    π_θ[nonzero_prob_actions, state] .-= (scale * minp)
    return π_θ
end

function _ensure_stochastic!(
    ::SoftmaxPolicy, π_f::Tabular, π_θ::AbstractMatrix, state; minp::F,
) where {F}
    # Actions with non-zero probability
    zero_prob_actions = π_θ_[:, state] .< minp
    if !any(zero_prob_actions)
        return π_θ
    end
    π_θ_ = softmax(π_θ)
    nonzero_prob_actions = (!).(zero_prob_actions)

    # Redistribute probability mass
    π_θ_[zero_prob_actions, state] .= minp
    π_θ_[nonzero_prob_actions, state] .-= (minp / length(zero_prob_actions))
    π_θ[:, state] .= log.(π_θ_)
    π_θ[:, state] .-= maximum(π_θ)

    return π_θ
end

function _ensure_stochastic(π::SimplexPolicy, π_f::Tabular, π_θ::NamedTuple, state; minp)
    f = x -> _ensure_stochastic!(π, π_f, x, state; minp)
    return treemap(f, π_θ)
end

function _ensure_stochastic(π::SoftmaxPolicy, π_f::Tabular, π_θ::NamedTuple, state; minp)
    f = x -> _ensure_stochastic!(π, π_f, x, state; minp)
    return treemap(f, π_θ)
end

function _ensure_stochastic(π, π_f::Tabular, π_θ::AbstractMatrix; minp::F) where {F}
    for state in 1:size(π_θ, 2)
        π_θ = _ensure_stochastic(π, π_f, π_θ; minp)
    end
    return π_θ
end


####################################################################
function assert_uniform(::SimplexPolicy, θ)
    out = treemap(θ -> all(sum(θ; dims=1) .== 1), θ)
    flag = true
    for key in keys(out)
        if !out[key]
            return false
        end
    end
    return true
end

function assert_uniform(::SoftmaxPolicy, θ)
    out = treemap(θ -> all(mapslices(φ -> φ .== φ[1], θ; dims=1)), θ)
    flag = true
    for key in keys(out)
        if !out[key]
            return false
        end
    end
    return true
end

####################################################################
function argmax_break_ties_randomly(rng::AbstractRNG, arr)
    inds = findall(x -> ==(maximum(arr))(x), arr)
    return length(inds) > 1 ? rand(rng, inds) : only(inds)
end
