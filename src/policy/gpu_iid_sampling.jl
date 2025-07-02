####################################################################
# GPU sampling
#
# A major issue with sampling from the GPU is that RNGs tend to live
# on the CPU. The below code is a work-around for this.
# See https://github.com/juliagpu/cuda.jl/issues/1480
####################################################################
function device_rng(seed, counter)
    rng = Random.default_rng()
    @inbounds Random.seed!(rng, seed, counter)
    rng
end

function barrier_rand!(D, A, seed, counter)
    A .= (x -> rand(device_rng(seed, counter), x)).(D)
end

# Note that this implementation is not a complete reflection of
function Random.rand!(rng::CUDA.RNG, D::AbstractArray{<:ContinuousUnivariateDistribution}, A)
    barrier_rand!(D, A, rng.seed, rng.counter)
    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    return A
end

function _sample(
    rng::CUDA.RNG, D::AbstractArray{<:ContinuousUnivariateDistribution}, sz::Int...;
    keepdims=true,
)
    if (length(sz) > 0 && sz[1] != 1) || (length(sz) > 0 && sz[1] == 1 && keepdims)
        A = CuArray{Float32}(undef, (size(D, 1), sz..., size(D)[begin+1:end]...))
        D = reshape(D, (size(D, 1), 1, size(D)[begin+1:end]...))
    else
        A = CuArray{Float32}(undef, size(D)...)
    end
    barrier_rand!(D, A, rng.seed, rng.counter)
    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    return A
end

function _sample(
    p::IIDPolicy{<:CuVector,D},
    rng::CUDA.RNG,
    model,
    model_θ,
    model_st,
    states::AbstractArray;
    num_samples=1,
    clip_actions=true,
) where {D}
    params, model_st = get_params(p, model, model_θ, model_st, states)

    actions = if ndims(params[1]) > 1
        # Sampling actions from a batch of states
        dist = p(params...)
        _sample(rng, dist, num_samples) # (action_dims, num_samples, batch_size)

    else
        # Sampling from a single state
        dist = p(params...)
        _sample(rng, dist, num_samples) # (action_dims, num_samples)
    end

    actions = convert.(eltype(eltype(p)), actions) # Required, since rand isn't type stable
    actions = clip_actions ? clip(p, actions) : actions

    return ChainRulesCore.ignore_derivatives(actions), model_st
end
####################################################################
