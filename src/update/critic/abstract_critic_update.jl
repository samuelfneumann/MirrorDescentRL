abstract type AbstractCriticUpdate <: AbstractUpdate end
abstract type AbstractActionValueCriticUpdate <: AbstractCriticUpdate end
abstract type AbstractStateValueCriticUpdate <: AbstractCriticUpdate end

function setup(
    up::AbstractCriticUpdate,
    π::AbstractPolicy,
    π_f,
    π_θ,
    π_st,
    vf::AbstractValueFunction,
    vf_f,
    vf_θ,
    vf_st,
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState
    error("setup not implemented for type $(typeof(up))")
end

function update(
    st::UpdateState{<:AbstractActionValueCriticUpdate},
    π::AbstractParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    qf::AbstractActionValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    qf_target_θ,
    qf_target_st,
    s_t,
    a_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    error("update not implemented for type $(typeof(st))")
end

# This version of `update` is for when no target nets are used.
function update(
    st::UpdateState{<:AbstractActionValueCriticUpdate},
    π::AbstractParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    vf::AbstractActionValueFunction,
    vf_f,
    vf_θ,
    vf_st,
    s_t,
    a_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    # TODO: will this be problematic if the first and second vf_st arguments are not
    # pointers? Then when updating, the value function state might become out-of-sync, since
    # we predict with it assuming it is a target net and assuming it is not a target net,
    # and these update re-assign the value function state, and point to different objects.
    #
    # For now this is okay, because we don't use any value function networks with state
    st, vf_θ, vf_st, _, π_st = update(
        st, π, π_f, π_θ, π_st, vf, vf_f, vf_θ, vf_st, vf_θ, vf_st, s_t, a_t, r_tp1,
        s_tp1, γ_tp1,
    )

    return st, vf_θ, vf_st, nothing, π_st
end

function update(
    st::UpdateState{<:AbstractStateValueCriticUpdate},
    π::AbstractParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    vf::AbstractStateValueFunction,
    vf_f,
    vf_θ,
    vf_st,
    vf_target_θ,
    vf_target_st,
    s_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    error("update not implemented for type $(typeof(st))")
end

# This version of `update` is for when no target nets are used.
function update(
    st::UpdateState{<:AbstractStateValueCriticUpdate},
    π::AbstractParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    vf::AbstractStateValueFunction,
    vf_f,
    vf_θ,
    vf_st,
    s_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    # TODO: will this be problematic if the first and second vf_st arguments are not
    # pointers? Then when updating, the value function state might become out-of-sync, since
    # we predict with it assuming it is a target net and assuming it is not a target net,
    # and these update re-assign the value function state, and point to different objects.
    #
    # For now this is okay, because we don't use any value function networks with state
    st, vf_θ, vf_st, _, π_st = update(
        st, π, π_f, π_θ, π_st, vf, vf_f, vf_θ, vf_st, vf_θ, vf_st, s_t, r_tp1, s_tp1, γ_tp1,
    )

    return st, vf_θ, vf_st, nothing, π_st
end
