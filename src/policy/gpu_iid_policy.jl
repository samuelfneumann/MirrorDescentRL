# TODO: delete this file after checking on GPU

#####################################################################
## Bounded Policy
#####################################################################
#function logprob_from_params(
#    p::BoundedPolicy{<:CuVector},
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
#    p::BoundedPolicy{<:CuVector},
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
#    p::BoundedPolicy{<:CuVector},
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
## UnBounded Policy
##
## TODO: there's a lot of code duplication between this block and the
## BoundedPolicy block above. I wonder if we could somehow combine
## these
#####################################################################
#function logprob_from_params(
#    p::UnBoundedPolicy{<:CuVector},
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
#    lp = (p |> logprob_function).(params..., samples)
#    return sum_ ? sum(lp; dims=1)[1, :, :] : lp
#end

#function logprob_from_params(
#    p::UnBoundedPolicy{<:CuVector},
#    actions::CuArray{F,2},
#    params::CuArray{F,2}...;
#    sum_=true,
#)::AbstractArray{F} where {F}
#    @info "calling logprob_from_params on gpu"
#    if size(params[1]) != size(actions)
#        error("must specify one set of parameters for each action")
#    end

#    samples = untransform(p, actions)
#    lp = (p |> logprob_function).(params..., samples)
#    return sum_ ? reshape(sum(lp; dims=1), :) : lp
#end

#function logprob_from_params(
#    p::UnBoundedPolicy{<:CuVector},
#    actions::CuArray{F,1},
#    params::CuArray{F,1}...;
#    sub,
#    sum_=true,
#)::AbstractArray{F} where {F}
#    @info "calling logprob_from_params on gpu"
#    if size(params[1]) != size(actions)
#        error("must specify one set of parameters for each action")
#    end

#    samples = untransform(p, actions)
#    lp = (p |> logprob_function).(params..., samples)
#    return sum_ ? [sum(lp)] : [lp]
#end
