struct DiscreteQ{F<:Function} <: AbstractActionValueFunction
    _reduct::F
end

function DiscreteQ(reduct::F=minimum) where {F}
    return DiscreteQ{F}(reduct)
end

continuous(::DiscreteQ) = false
discrete(::DiscreteQ) = true

function predict(
    q::DiscreteQ,
    model::Tabular,
    model_θ::NamedTuple,
    model_st::NamedTuple,
)
    return model(model_θ, model_st)
end

function predict(
    q::DiscreteQ,
    model::Tabular,
    model_θ,
    model_st,
    state::Int;
)
    return predict(q, model, model_θ, model_st, [state]; reduct=false)
end

function predict(
    q::DiscreteQ,
    model::Tabular,
    model_θ,
    model_st,
    state::Int,
    action::Int;
)
    q, model_st = predict(q, model, model_θ, model_st, [state]; reduct=false)
    return q[action], model_st
end

function predict(
    q::DiscreteQ,
    model::Tabular,
    model_θ,
    model_st,
    states::Vector{Int},
    actions::Vector{Int};
)
    q, model_st = predict(q, model, model_θ, model_st, states; reduct=false)
    return q[[CartesianIndex(actions[i], i) for i in 1:length(states)]], model_st
end

function predict(
    q::DiscreteQ,
    model::Tabular,
    model_θ,
    model_st,
    states::Matrix{Int},
    actions::Matrix{Int};
)
    @assert size(states, 1) == 1
    @assert size(actions, 1) == 1
    q, model_st = predict(q, model, model_θ, model_st, states; reduct=false)
    return q[[CartesianIndex(actions[1, i], i) for i in 1:length(states)]], model_st
end

function predict(
    q::DiscreteQ,
    model,
    model_θ,
    model_st,
    state::AbstractArray{F,1};
    reduct=true,
) where {F}
    state = reshape(state, :, 1)
    return predict(q, model, model_θ, model_st, state; reduct=reduct)
end

function predict(
    q::DiscreteQ,
    model,
    model_θ,
    model_st,
    state::AbstractArray{F,2};
    reduct=true,
) where {F}
    out, model_st = model(state, model_θ, model_st)
    n = num_approx(q, out)

    if reduct && n != 1
        last = ndims(out)
        out = dropdims(q._reduct(out; dims=last); dims=last)
    end

    return out, model_st
end

function predict(
    q::DiscreteQ,
    model,
    model_θ,
    model_st,
    state::AbstractArray{FS,2},
    action::AbstractArray{FA,2};
    reduct=true,
) where {FS, FA}
    return predict(
        q, model, model_θ, model_st, state, dropdims(action; dims=1);
        reduct=reduct,
    )
end

function predict(
    q::DiscreteQ,
    model,
    model_θ,
    model_st,
    state::AbstractArray{FS,2},
    action::AbstractArray{FA,1};
    reduct=true,
) where {FS, FA}
    values, model_st = predict(q, model, model_θ, model_st, state; reduct=reduct)
    bs = batch_size(q, values)
    if reduct
        return values[[CartesianIndex(action[i], i) for i in 1:bs]], model_st
    else
        n = num_approx(q, values)
        if n == 1
            indices = [
                CartesianIndex(action[i], i)
                for i in 1:bs
            ]
            return values[indices], model_st
        else
            return reshape(
                values[[
                    CartesianIndex(action[i], i, j)
                    for j in 1:n, i in 1:bs
                ]],
                size(values)[2:end],
            ), model_st
        end
    end
end

####################################################################

# Provides the function approximator with a tuple of inputs: (state, action)
struct Q{F<:Function} <: AbstractActionValueFunction
    _reduct::F
end

function Q(reduct::F=minimum) where {F}
    return new{F}(reduct)
end

continuous(::Q) = true
discrete(::Q) = false

function predict(
    q::Q,
    model,
    model_θ,
    model_st,
    state_action;
    reduct=true,
)
    if length(state_action) != 2
        len = length(state_action)
        error("expected input to have length 2 (state, action) but got $(len)")
    end
    return predict(
        q, model, model_θ, model_st, state_action[1], state_action[2]; reduct=reduct,
    )
end

function predict(
    q::Q,
    model,
    model_θ,
    model_st,
    state::AbstractArray{FS,2},
    action::AbstractArray{FA,2};
    reduct=true,
) where {FS, FA}
    # Outputs are (ndims, batch, napprox)
    out, model_st = model((state, action), model_θ, model_st)
    n = num_approx(q, out)

    if size(out)[1] != 1
        # Ensure the function approximators only output a single action-value
        error("expected network to output one q-value but got $(size(out)[1])")
    end

    if reduct && n != 1
        # Reduce over the predictions from each approximator
        last = length(size(out))
        out = dropdims(q._reduct(out; dims=last); dims=last)
    end

    # Squeeze the number of action-value predictions, which is always 1
    return dropdims(out, dims=1), model_st
end

# Outputs are either (action_dims × batch_size × num_approx) or (num_actions × batch_size)
# or (num_actions)
function num_approx(::Union{Q,DiscreteQ,V}, out)::Int
    if ndims(out) <= 2
        return 1
    elseif ndims(out) == 3
        return size(out)[end]
    else
        error("ndims(out) should be <= 3 but got $(ndims(out))")
    end
end

function batch_size(::Union{Q,DiscreteQ,V}, out)::Int
    if ndims(out) == 1
        return 1
    elseif 2 <= ndims(out) <= 3
        return size(out)[2]
    else
        error("ndims(out) should be <= 3 but got $(ndims(out))")
    end
end
