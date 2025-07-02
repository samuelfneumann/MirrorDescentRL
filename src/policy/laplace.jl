function rsample_with_logprob(
    p::UnBoundedPolicy{V,Laplace},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray; # 2-dimensional
    num_samples = 1,
    clip_actions = false,
) where {V<:AbstractArray}
    actions, model_st, μ, θ = _rsample(
        p, rng, model, model_θ, model_st, states;
        num_samples=num_samples, clip_actions=clip_actions,
    )

    lp = laplacelogpdf.(μ, θ, actions)
    lp = sum(lp; dims=1)[1, :, :]

    return actions, lp, model_st
end

function rsample(
    p::UnBoundedPolicy{V,Laplace},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray; # 2-dimensional
    num_samples = 1,
    clip_actions = false,
) where {V<:AbstractArray}
    actions, model_st, _, _ = _rsample(
        p, rng, model, model_θ, model_st, states;
        num_samples=num_samples, clip_actions=clip_actions,
    )
    return actions, model_st
end

function _rsample(
    p::UnBoundedPolicy{V,Laplace},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray; # 2-dimensional
    num_samples=1,
    clip_actions=false,
) where {V<:AbstractArray}
    params, model_st = get_params(p, model, model_θ, model_st, states)
    μ, θ = params
    F = eltype(V)

    actions = if ndims(μ) > 1
        # Sampling actions from a batch of states
        batch_size = size(μ)[end]

        # If in training mode, sample actions from the distribution in each state
        ε = ChainRulesCore.ignore_derivatives(
            randexp(rng, F, (size(μ, 1), num_samples, batch_size)),
        ) .* ifelse.(rand(rng, Bool, size(μ, 1), num_samples, batch_size), 1, -1)
        μ = reshape(μ, size(μ, 1), 1, batch_size)
        θ = reshape(θ, size(μ, 1), 1, batch_size)
        μ .+ ε .* θ
    else
        # Sampling actions from a single state
        ε = ChainRulesCore.ignore_derivatives(
            randexp(rng, F, (size(μ, 1), num_samples)),
        ) .* ifelse.(rand(rng, Bool, size(μ, 1), num_samples), 1, -1)
        μ .+ ε .* θ
    end

    return actions, model_st, μ, θ
end
