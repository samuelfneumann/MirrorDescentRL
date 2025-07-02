abstract type AbstractActorUpdate <: AbstractUpdate end
abstract type AbstractPolicyGradientStyleUpdate <: AbstractUpdate end

function setup(
    up::AbstractActorUpdate,
    π::AbstractPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)
    error("setup not implemented for type $(typeof(up))")
end

function setup(
    up::AbstractPolicyGradientStyleUpdate,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG,
)
    error("setup not implemented for type $(typeof(up))")
end

function update(
    st::UpdateState{<:AbstractActorUpdate},
    π::AbstractPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    states,
)
    error("update not implemented for type $(typeof(st))")
end

function update(
    st::UpdateState{<:AbstractPolicyGradientStyleUpdate},
    π::AbstractParameterisedPolicy,
    π_f,    # Actor policy model
    π_θ,    # Actor policy model parameters
    π_st,   # Actor policy model state
    s_t::AbstractArray, # Must be >= 2D
    A_t::AbstractVector, # Advantage estimate in each state
    γ_t::AbstractVector,
)
    error("update not implemented for type $(typeof(st))")
end
