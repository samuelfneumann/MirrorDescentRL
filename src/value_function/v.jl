struct V{F<:Function} <: AbstractStateValueFunction
    _reduct::F
end

function V(reduct::F=minimum) where {F}
    return V{F}(reduct)
end

continuous(::V) = true
discrete(::V) = true

function predict(
    v::V,
    model,
    model_θ,
    model_st,
    state::AbstractArray{F,1};
    reduct=true,
) where {F}
    state = reshape(state, :, 1)
    out, model_st = predict(v, model, model_θ, model_st, state; reduct=reduct)
    return out, model_st
end

function predict(
    v::V,
    model,
    model_θ,
    model_st,
    state::AbstractArray{F,2};
    reduct=true,
) where {F}
    out, model_st = model(state, model_θ, model_st)
    n = num_approx(v, out)

    if reduct && n != 1
        last = ndims(out)
        out = dropdims(v._reduct(out; dims=last); dims=last)
    end

    return dropdims(out; dims=1), model_st
end
####################################################################
