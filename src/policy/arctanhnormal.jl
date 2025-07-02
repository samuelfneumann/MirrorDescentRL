####################################################################
# Consants
####################################################################
const _ARCTANH_NORMAL_EPSILON = 1f-6
####################################################################

####################################################################
# Sampling
#
# Override sampling so that clip_actions is false
####################################################################
function sample(
    p::BoundedPolicy{F,ArctanhNormal}, rng::AbstractRNG, args...;
    clip_actions=false, kwargs...,
) where {F}
    return _sample(p, rng, args...; clip_actions=clip_actions, kwargs...)
end

function sample_with_logprob(
    p::BoundedPolicy{F,ArctanhNormal},
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
    p::BoundedPolicy{<:AbstractVector,ArctanhNormal},
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
        normlogpdf.(μ, σ, x_t) .-
        log1p.(-(y_t .^ 2) .+ _ARCTANH_NORMAL_EPSILON) .-
        log.(p._action_scale)
    )

    # Sum log probabilities along action dimensions, do get the full log probability for the
    # multi-dimensional IID policy distribution
    lp = sum(lp; dims=1)[1, :, :]

    return actions, lp, model_st
end

function rsample(
    p::BoundedPolicy{<:AbstractVector,ArctanhNormal},
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
    p::BoundedPolicy{<:AbstractVector,ArctanhNormal},
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

    y_t = tanh.(x_t)
    actions = transform(p, y_t)
    actions = clip_actions ? clip(p, actions) : actions

    return actions, model_st, x_t, y_t, μ, σ
end
####################################################################

#####################################################################
## GPU Implementation of Log Density
##
## TODO: eventually, we can actually use these implementations for
## both CPU and GPU, because it will work on both devices.
##
## BAD: this is copy-pasted from gpu_iid_policy.jl. Addressing the
## TODO above and the TODO in gpu_iid_policy.jl will allow us to get
## rid of this code duplication
#####################################################################
#function logprob_from_params(
#    p::BoundedPolicy{<:CuVector,ArctanhNormal},
#    actions::CuArray{F,3},
#    params::CuArray{F,2}...;
#    sum_=true,
#)::AbstractArray{F} where {F}
#    @info "calling logprob_from_params on gpu"
#    if size(params[1]) != size(actions)[[1, 3]]
#        error("must specify one set of parameters for each action")
#    end

#    samples = untransform(p, actions)
#    params = reshape.(params, size.(params, 1), 1, size.(params, 2))
#    lp = (p |> logprob_function).(params..., samples) .- log.(p._action_scale)
#    return sum_ ? sum(lp; dims=1)[1, :, :] : lp
#end

#function logprob_from_params(
#    p::BoundedPolicy{<:CuVector,ArctanhNormal},
#    actions::CuArray{F,2},
#    params::CuArray{F,2}...;
#    sum_=true,
#)::AbstractArray{F} where {F}
#    @info "calling logprob_from_params on gpu"
#    if size(params[1]) != size(actions)
#        error("must specify one set of parameters for each action")
#    end

#    samples = untransform(p, actions)
#    lp = (p |> logprob_function).(params..., samples) .- log.(p._action_scale)
#    return sum_ ? reshape(sum(lp; dims=1), :) : lp
#end

#function logprob_from_params(
#    p::BoundedPolicy{<:CuVector,ArctanhNormal},
#    actions::CuArray{F,1},
#    params::CuArray{F,1}...;
#    sum_=true,
#)::AbstractArray{F} where {F}
#    @info "calling logprob_from_params on gpu"
#    if size(params[1]) != size(actions)
#        error("must specify one set of parameters for each action")
#    end

#    samples = untransform(p, actions)
#    lp = (p |> logprob_function).(params..., samples) .- log.(p._action_scale)
#    return sum_ ? [sum(lp)] : [lp]
#end
#####################################################################
