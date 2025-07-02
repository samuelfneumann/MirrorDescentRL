####################################################################
# Consants
####################################################################
const _LOGIT_NORMAL_EPSILON = 1f-6
####################################################################

####################################################################
# Sampling
####################################################################
function sample_with_logprob(
    p::BoundedPolicy{F,LogitNormal},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray; # 2-dimensional
    num_samples=1,
    clip_actions=false,
) where {F}
    actions, lnπ, model_st = rsample_with_logprob(
        p, rng, model, model_θ, model_st, states;
        num_samples = num_samples, clip_actions = clip_actions,
    )
    return ChainRulesCore.ignore_derivatives(actions), lnπ, model_st
end

function rsample_with_logprob(
    p::BoundedPolicy{<:AbstractVector,LogitNormal},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray; # 2-dimensional
    num_samples=1,
    clip_actions=false,
)
    actions, model_st, x_t, y_t, μ, σ = _rsample(
        p, rng, model, model_θ, model_st, states;
        num_samples=num_samples, clip_actions=clip_actions,
    )

    lp = (
        normlogpdf.(μ, σ, x_t) .+
        inv.(y_t .* (1 .- y_t) .+ _LOGIT_NORMAL_EPSILON) .-
        log.(p._action_scale)
    )
    lp = sum(lp; dims=1)[1, :, :]

    return actions, lp, model_st
end

function rsample(
    p::BoundedPolicy{<:AbstractVector,LogitNormal},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray; # 2-dimensional
    num_samples=1,
    clip_actions=false,
)
    actions, model_st, _, _, _, _ = _rsample(
        p, rng, model, model_θ, model_st, states;
        num_samples=num_samples, clip_actions=clip_actions,
    )
    return actions, model_st
end

function _rsample(
    p::BoundedPolicy{<:AbstractVector,LogitNormal},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray; # 2-dimensional
    num_samples=1,
    clip_actions=false,
)
    params, model_st = get_params(p, model, model_θ, model_st, states)
    μ, σ = params
    F = eltype(eltype(p))

    x_t = if ndims(μ) > 1
        # Sampling actions from a batch of states
        batch_size = size(μ)[end]

        # If in training mode, sample actions from the distribution in each state
        ε = ChainRulesCore.ignore_derivatives(
            randn(rng, F, (size(μ, 1), num_samples, batch_size)),
        )
        μ = reshape(μ, size(μ, 1), 1, batch_size)
        σ = reshape(σ, size(μ, 1), 1, batch_size)
        μ .+ ε .* σ
    else
        # Sampling actions from a single state
        ε = ChainRulesCore.ignore_derivatives(randn(rng, F, (size(μ, 1), num_samples)))
        μ .+ ε .* σ
    end

    y_t = logistic.(x_t)
    actions = transform(p, y_t)
    actions = clip_actions ? clip(p, actions) : actions

    return actions, model_st, x_t, y_t, μ, σ
end
####################################################################
