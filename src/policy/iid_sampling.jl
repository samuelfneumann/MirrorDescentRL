####################################################################
# sample
#
# By default, always clip actions when sampling. This is because
# we don't need to worry about stopping gradients when sampling, as
# opposed to rsampling. Furthermore, with clip set to true, we always
# clip BoundedPolicy samples to stay within the distribution support,
# but for UnBoundedPolicy we only clip depending on the internals
# of the UnBoundedPolicy
####################################################################
# Trick so that we can override default sampling arguments easier
function sample(p::IIDPolicy{V,D}, rng::AbstractRNG, args...; kwargs...) where {V,D}
    return _sample(p, rng, args...; kwargs...)
end

# Generic sample while returning the logprob of sampled actions
function sample_with_logprob(
    p::IIDPolicy,
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples=1,
    clip_actions=true,
)
    actions, model_st = _sample(
        p, rng, model, model_θ, model_st, states;
        num_samples = num_samples, clip_actions = clip_actions,
    )
    return actions, logprob(p, model, model_θ, model_st, states, actions)...
end

function _sample(
    p::IIDPolicy{V,D},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples=1,
    clip_actions=true,
) where {V,D}
    params, model_st = get_params(p, model, model_θ, model_st, states)

    actions = if ndims(params[1]) > 1
        # Sampling actions from a batch of states
        batch_size = size(params[1])[end]
        dist = p(params...)
        actions = stack([rand.(rng, dist) for _ in 1:num_samples]; dims=2)

    else
        # Sampling from a single state
        dist = p(params...)
        out = reduce(hcat, rand.(rng, dist, num_samples))

        # Reshape to be (action_dims, num_samples)
        transpose(out)
    end

    actions = convert.(eltype(V), actions) # Required, since rand isn't type stable
    actions = clip_actions ? clip(p, actions) : actions
    return ChainRulesCore.ignore_derivatives(actions), model_st
end
####################################################################

####################################################################
# Methods to reparameterize and sample
#
# We always fall back to rsample through quantile functions. This is
# because this operation is differentiable with CUDA, and it is a
# viable way to rsample. But, for some distributions, we can rsample
# in a smarter manner (e.g. ArctanhNormal, LogitNormal, Laplace,
# Normal, etc.) and so those implementations are separated into
# their own respective files.
#
# Many distributions don't have "special" ways to rsample, but can
# be rsampled through their quantile function. E.g. the Kumaraswamy
# distribution has this property. Hence, we always fall back to this
# default behaviour.
####################################################################
function rsample(
    p::IIDPolicy, rng::AbstractRNG, args...; kwargs...,
)
    return _rsample_through_quantile(p, rng, args...; kwargs...)
end

function rsample_with_logprob(
    p::IIDPolicy, rng::AbstractRNG, args...; kwargs...,
)
    return _rsample_through_quantile_with_logprob(p, rng, args...; kwargs...)
end

function _rsample_through_quantile(
    p::IIDPolicy{V,D},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples=1,
    clip_actions=true,
) where {V,D}
    # For many distributions, rand is differentiable. Should we just try to use rand, or use
    # this method here where we sample u ~ U(0, 1) and reflect through the quantile
    # function.
    params, model_st = get_params(p, model, model_θ, model_st, states)
    F = eltype(eltype(p))

    samples = if ndims(params[1]) > 1
        # Sampling actions from a batch of states
        batch_size = size(params[1])[end]
        action_size = size(params[1], 1)
        # @assert ndims(params[1]) == 2

        # Reshape for broadcasting
        reshaped_params = map(x -> reshape(x, action_size, 1, batch_size), params)
        y = ChainRulesCore.@ignore_derivatives rand(
            rng, Float32, action_size, num_samples, batch_size,
        )
        (p |> quantile_function).(reshaped_params..., y)
    else
        # Sampling actions from a single state
        action_size = size(params[1], 1)
        y = ChainRulesCore.@ignore_derivatives rand(rng, Float32, action_size, num_samples)
        (p |> quantile_function).(params..., y)
    end

    actions = transform(p, samples)
    actions = clip_actions ? clip(p, actions) : actions

    return actions, model_st
end

function _rsample_through_quantile_with_logprob(
    p::IIDPolicy{V,D},
    rng::AbstractRNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples=1,
    clip_actions=true,
) where {V,D}
    # For many distributions, rand is differentiable. Should we just try to use rand, or use
    # this method here where we sample u ~ U(0, 1) and reflect through the quantile
    # function.
    params, model_st = get_params(p, model, model_θ, model_st, states)
    F = eltype(eltype(p))

    samples, reshaped_params = if ndims(params[1]) > 1
        # Sampling actions from a batch of states
        batch_size = size(params[1])[end]
        action_size = size(params[1], 1)

        # Reshape for broadcasting
        reshaped_params = map(x -> reshape(x, action_size, 1, batch_size), params)

        y = ChainRulesCore.@ignore_derivatives rand(
            rng, Float32, action_size, num_samples, batch_size,
        )
        samples = (p |> quantile_function).(reshaped_params..., y)

        samples, reshaped_params
    else
        # Sampling actions from a single state
        action_size = size(params[1], 1)
        y = ChainRulesCore.@ignore_derivatives rand(
            rng, Float32, action_size, num_samples,
        )
        (p |> quantile_function).(params..., y), params
    end

    actions = transform(p, samples)
    actions = clip_actions ? clip(p, actions) : actions

    # Transform (possibly clipped) actions back to samples, so that we get the correct
    # log-probability for the sampled actions
    samples = clip_actions ? untransform(p, actions) : samples
    lp = (p |> logprob_function).(reshaped_params..., samples)

    return actions, lp, model_st
end
