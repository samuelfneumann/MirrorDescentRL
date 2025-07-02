module ActorCritic

import ChoosyDataLoggers
import ChoosyDataLoggers: @data
function __init__()
    ChoosyDataLoggers.@register
end

import MLDataDevices: AbstractDevice
import Reproduce: @param_from # For construction utils
using MLUtils: unsqueeze
using NNlib
using ChainRulesCore
using AbstractTrees
using ExtendedDistributions
using DistributionsAD
using LinearAlgebra
using Lux
using CUDA
using LuxCUDA
using Random
using Roots
using SparseArrays
using StatsBase
using Zygote
using Adapt
using Optimisers
using Tullio
using Lazy


# Feature constructors
include("util/feature/feature.jl")

# Function Approximators
export Linear, Tabular
include("util/approximator/linear.jl")
include("util/approximator/tabular.jl")
include("util/approximator/lux.jl")
include("util/approximator/util.jl")

# GPU Utilities
include("util/gpu.jl")

include("env/environment.jl")
include("policy/policy.jl")
include("value_function/value_function.jl")
include("update/update.jl")

# Experience Replay
include("util/buffer/buffer.jl")

export
    OnlineQAgent,
    PGFunctionalBaselineAgent,
    PGAgent,
    OnlinePGAgent,
    BatchQAgent,
    BatchQAgentMDTest,
    UpdateRatio,
    AbstractAgentWrapper,
    AbstractAgentActionWrapper,
    RandomFirstActionAgent

include("agent/abstract_agent.jl")
include("agent/agent_wrapper.jl")
include("agent/update_ratio.jl")
include("agent/online_qagent.jl")
include("agent/pg_agent.jl")
include("agent/online_pg_agent.jl")
include("agent/pg_functional_baseline_agent.jl")
include("agent/batch_qagent.jl")

include("util/episode.jl")

# General construction utilities: updates, optimisers, buffers, policies, value functions
include("util/construct.jl")

# Experiment Utils
include("util/exp.jl")
include("util/exp/brax_experiment.jl")
include("util/exp/gymnasium_experiment.jl")
include("util/exp/default_experiment.jl")
include("util/exp/simplex_experiment.jl")

end # module ActorCritic
