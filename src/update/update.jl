using Optimisers
using ChainRulesCore

abstract type AbstractUpdate end

function update(up::AbstractUpdate, args...; kwargs...)
    error(
        "update not implemented for type $(typeof(up)) with args $(typeof(args)) " *
        "and kwargs $(typeof(kwargs))"
    )
end

# Each AbstractUpdate needs to override this
function setup(up::AbstractUpdate, args...; kwargs...)
    error(
        "setup not implemented for type $(typeof(up)) with args $(typeof(args)) " *
        "and kwargs $(typeof(kwargs))"
    )
end

export
    # Abstract types and UnionAlls
    AbstractActorUpdate,
    ClosedFormMirrorDescentUpdate,
    ProximalMDUpdate,
    KLPenaltyUpdate,

    NullActorUpdate,
    PPO,
    REINFORCE,

    # MPO
    ContinuousMPO,
    ContinuousProximalMDMPO,
    DiscreteMPO,
    SimplexMDMPO,
    SimplexMPO,
    DiscreteProximalMDMPO,
    SimplexProximalMDMPO,

    # CCEM
    ContinuousCCEM,
    ContinuousProximalMDCCEM,
    DiscreteCCEM,
    SimplexCCEM,
    SimplexMDCCEM,
    DiscreteProximalMDCCEM,
    SimplexProximalMDCCEM,

    # FKL to Boltzmann over action values
    ContinuousFKL,
    DiscreteFKL,

    # RKL to Boltzmann over action values
    ContinuousRKL,
    ContinuousRKLKL,
    ContinuousProximalMDRKL,
    DiscreteRKL,
    SimplexRKL,
    SimplexMDRKL,
    DiscreteProximalMDRKL,
    SimplexProximalMDRKL,

    SimplexPG,
    OnPolicySimplexPG,
    SimplexPMD,
    SimplexSPMD,
    UpdateState

export AbstractCriticUpdate, Sarsa, TD

export AbstractBellmanRegulariser,
       EntropyBellmanRegulariser,
       KLBellmanRegulariser,
       NullBellmanRegulariser

struct UpdateState{U, O<:Union{Nothing,Optimisers.AbstractRule}, S<:NamedTuple}
    _update::U
    _optim::O
    _state::S
end

function replace(u::UpdateState; kwargs...)
    UpdateState(u._update, u._optim, merge(u._state, kwargs))
end

include("critic/abstract_critic_update.jl")
include("actor/abstract_actor_update.jl")

function setup(
    up::AbstractCriticUpdate,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    optim::Optimisers.AbstractRule,
    seed::Integer,
)::UpdateState{<:AbstractCriticUpdate}
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    return setup(up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, optim, rng)
end

function setup(
    up::AbstractActorUpdate,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim,
    seed::Integer,
)::UpdateState{<:AbstractActorUpdate}
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    return setup(up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, optim, rng)
end

abstract type AbstractRegulariser end

struct RegulariserState{T<:AbstractRegulariser, S<:NamedTuple}
    _reg::T
    _state::S
end

# Critic updates
include("critic/regularisation.jl")
include("critic/sarsa.jl")
include("critic/td.jl")

# Actor updates
include("actor/null_update.jl")
include("actor/reinforce.jl")
include("actor/ppo.jl")

# Continuous
include("actor/continuous/mpo.jl")
include("actor/continuous/proximal_mdmpo.jl")
# Discrete
include("actor/discrete/mpo.jl")
include("actor/discrete/proximal_mdmpo.jl")
include("actor/discrete/simplex_proximal_mdmpo.jl")
include("actor/discrete/simplex_mdmpo.jl")
include("actor/discrete/simplex_mpo.jl")

# Continuous
include("actor/continuous/proximal_mdccem.jl")
include("actor/continuous/ccem.jl")
# Discrete
include("actor/discrete/simplex_mdccem.jl")
include("actor/discrete/proximal_mdccem.jl")
include("actor/discrete/ccem.jl")
include("actor/discrete/simplex_proximal_mdccem.jl")
include("actor/discrete/simplex_ccem.jl")

# Continuous
include("actor/continuous/fkl.jl")
# Discrete
include("actor/discrete/fkl.jl")

# Continuous
include("actor/continuous/rkl.jl")
include("actor/continuous/rkl_kl.jl")
include("actor/continuous/proximal_mdrkl.jl")
# Discrete
include("actor/discrete/rkl.jl")
include("actor/discrete/proximal_mdrkl.jl")
include("actor/discrete/simplex_proximal_mdrkl.jl")
include("actor/discrete/simplex_mdrkl.jl")
include("actor/discrete/simplex_rkl.jl")

include("actor/discrete/simplex_pg.jl")
include("actor/discrete/simplex_onpolicypg.jl")
include("actor/discrete/simplex_pmd.jl")
include("actor/discrete/simplex_spmd.jl")

# All proximal update types
const ProximalMDUpdate = Union{
    # CCEM
    ContinuousProximalMDCCEM,
    DiscreteProximalMDCCEM,
    SimplexProximalMDCCEM,

    # MPO
    ContinuousProximalMDMPO,
    DiscreteProximalMDMPO,
    SimplexProximalMDMPO,

    # RKL
    SimplexProximalMDRKL,
    DiscreteProximalMDRKL,
    ContinuousProximalMDRKL,
}

const KLPenaltyUpdate = Union{
    ContinuousRKLKL,
    <:ProximalMDUpdate,
}

const ClosedFormMirrorDescentUpdate = Union{
    SimplexMDCCEM, SimplexMDRKL, SimplexMDMPO
}
