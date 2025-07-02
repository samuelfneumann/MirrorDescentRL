abstract type AbstractValueFunction end

abstract type AbstractActionValueFunction <: AbstractValueFunction end
abstract type AbstractStateValueFunction <: AbstractValueFunction end

# TODO: we have AbstractDiscreteParameterizedPolicy and
# AbstractContinuousParameterizedPolicy types, should we do the same for value functions?
# And then update the value function update interfaces?
#
# This would be hard because some value function implementations (e.g. V) work for both
# continuous and discrete

abstract type AbstractTabularValueFunction <: AbstractValueFunction end

(vf::AbstractValueFunction)(args...; kwargs...) = predict(vf, args...; kwargs...)

function reduct(::AbstractValueFunction)::Function
    error("reduct not implemented")
end

function continuous(::AbstractValueFunction)::Bool
    error("continuous not implemented")
end

function discrete(::AbstractValueFunction)::Bool
    error("discrete not implemented")
end

"""
    predict(
        q::DiscreteQ,
        model,
        model_θ,
        model_st,
        state::AbstractArray{F,2},
        [action::AbstractArray{FA,1};
        reduct=true
    ) where {FS, FA}

    predict(q::Q, model, model_θ, model_st, state_action; reduct=true)

    predict(
        q::Q,
        model,
        model_θ,
        model_st,
        state::AbstractArray{FS,2},
        action::AbstractArray{FA,2};
        reduct=true
    ) where {FS, FA}

Predict the state or action value. The inputted states `state` must always be a matrix, with
features along the columns (dimension 1) and states in the batch along the rows (dimension
2).

## Continuous Action-Value Functions: `Q`
For `Q` types, which predict action values for continuous actions, the actions must always
be specified. Similarly to the `state` parameter, the `action` parameter is a matrix, with
action dimensions along the columns (dimension 1) and subsequent actions in the batch along
the rows (dimension 2).

## Discrete Action-Value Functions: `DiscreteQ`
For `DiscreteQ` types, which predict action values for discrete actions, we have two
choices. If no action is inputted, then the values for all actions for all states in the
batch are returned as a matrix. If actions are specified using the `action` argument, which
must be a vector, then the value for each of the specified actions is returned for the
corresponding state.
"""
function predict end
