struct NullActorUpdate <: AbstractActorUpdate end

function setup(
    up::NullActorUpdate,
    ::AbstractEnvironment,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim,
    rng,
)::UpdateState{NullActorUpdate}
    return UpdateState(up, optim, NamedTuple())
end

function update(
    st::UpdateState{NullActorUpdate},
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    qf::AbstractActionValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    return st, π_θ, π_st, qf_st
end
