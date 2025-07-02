module DefaultExperimentUtils

using ExtendedDistributions
using NNlib
using Lux
using Random
using ..ActorCritic
import Reproduce: @param_from

const LOG_STD_THRESHOLD = log(1000f0)

####################################################################
# Environment Construction
####################################################################
const ClassicControlEnvironment = Union{
    Type{<:Pendulum},
    Type{<:MountainCar},
    Type{<:Acrobot},
    Type{<:Cartpole},
}

function construct_env(config, rng)
    @param_from type config

    type = getproperty(ActorCritic, Symbol(type))
    env = construct_env(type, config, rng)

    if "wrapper" in keys(config)
        @param_from wrapper config
        @param_from type wrapper
        type = getproperty(ActorCritic, Symbol(type))
        env = construct_env_wrapper(type, wrapper, env, deepcopy(rng))
    end
    return env
end

# Fallback
function construct_env(type, config, rng)
    @param_from kwargs config
    ks = keys(kwargs)
    vs = values(kwargs)
    keywords = [Symbol(k) => v for (k, v) in zip(Symbol.(ks), vs)]

    continuous = get(kwargs, "continuous", false)
    env = if continuous
        ClipAction(type(rng; keywords...))
    else
        type(rng; keywords...)
    end

    cutoff_episodes = "step_limit" in keys(config)
    return if cutoff_episodes
        @param_from step_limit config
        StepLimit(env, step_limit)
    else
        env
    end
end


function construct_env_wrapper(::Type{<:RewardNoise}, config, env, rng)
    @param_from p config

    @param_from dist config
    dist = get_dist(dist)

    return RewardNoise(env, p, dist, rng)
end

function get_dist(dist_config)
    @param_from type dist_config
    dist_type = getproperty(ExtendedDistributions, Symbol(type))

    args = if "args" in keys(dist_config)
        @param_from args dist_config
    else
        tuple()
    end

    kwargs = if "kwargs" in keys(dist_config)
        @param_from kwargs dist_config
        ks = keys(kwargs)
        vs = values(kwargs)
        [Symbol(k) => v for (k, v) in zip(Symbol.(ks), vs)]
    else
        NamedTuple()
    end

    return dist_type(args...; kwargs...)
end

function construct_env(type::Type{<:Bimodal}, config, rng)
    return StepLimit(Bimodal(), 1)
end

function construct_env(type::Type{<:TwoArmBandit}, config, rng)
    @param_from kwargs config
    @param_from Δ kwargs
    @param_from σ kwargs
    @param_from γ kwargs
    env = TwoArmBandit(rng; γ, Δ, σ)

    cutoff_episodes = "step_limit" in keys(config)
    return if cutoff_episodes
        @param_from step_limit config
        StepLimit(env, step_limit)
    else
        env
    end
end

function construct_env(type::Type{<:AliasedState}, config, rng)
    @param_from kwargs config
    ks = keys(kwargs)
    vs = values(kwargs)
    keywords = [Symbol(k) => v for (k, v) in zip(Symbol.(ks), vs)]

    return AliasedState(rng; keywords...)
end

function construct_env(type::Union{Type{<:CliffWorld},Type{<:NoisyCliffWorld}}, config, rng)
    @param_from kwargs config
    ks = keys(kwargs)
    vs = values(kwargs)
    keywords = [Symbol(k) => v for (k, v) in zip(Symbol.(ks), vs)]

    env = type(rng; keywords...)

    cutoff_episodes = "step_limit" in keys(config)
    return if cutoff_episodes
        @param_from step_limit config
        StepLimit(env, step_limit)
    else
        env
    end
end
####################################################################

####################################################################
# Policy construction
####################################################################
function construct_policy_approximator(config, env, rng; feature_creator=nothing)
    @param_from type config
    @param_from approx config
    @param_from type config

    return construct_policy_approximator(approx, Symbol(type), env, rng; feature_creator)
end

function construct_policy_approximator(
    config, policy::Symbol, env, rng; feature_creator,
)
    @param_from type config
    approx_type = Symbol(type)
    return construct_policy_approximator(
        Val(approx_type), policy, config, env, rng; feature_creator,
    )
end

function construct_policy_approximator(
    ::Val{:Tabular}, policy::Symbol, config, env, rng; feature_creator,
)
    msg = "tabular parameterization only supports softmax and simplex policies"
    @assert (policy == :SoftmaxPolicy || policy == :SimplexPolicy) msg

    @param_from init config
    init = if occursin(".", init)
        init = split(init, ".")
        getproperty(eval(Symbol(init[1])), Symbol(init[2]))
    else
        eval(Symbol(init))
    end

    n_actions = action_space(env).n
    n_states = observation_space(env).n

    return Tabular(n_states[1], n_actions[1]; init)
end

function construct_policy_approximator(
    ::Val{:Lux}, policy::Symbol, config, env, rng; feature_creator,
)
    in_size = if feature_creator != nothing
        obs = rand(Lux.replicate(rng), observation_space(env))
        size(feature_creator(obs), 1)
    else
        size(ActorCritic.observation_space(env), 1)
    end

    out_size = if ActorCritic.continuous(ActorCritic.action_space(env))
        size(ActorCritic.action_space(env), 1)
    else
        ActorCritic.action_space(env).n[1]
    end

    return construct_policy_approximator(
        Val(:Lux), policy, in_size, out_size, config, env, rng,
    )
end

function construct_policy_approximator(
    ::Val{:Lux}, policy::Symbol, in_size, out_size, config, env, rng,
)
    @param_from hidden config
    @param_from act config
    @assert length(hidden) == length(act)

    @param_from init config
    init = getproperty(Lux, Symbol(init))
    last_layer_init = if "last_layer_init" in keys(config)
        @param_from last_layer_init config
        getproperty(Lux, Symbol(last_layer_init))
    else
        init
    end

    paths = []
    final_activation = final_act(Val(policy), env)
    layer_sizes = [in_size, hidden..., out_size]
    for i in 1:n_paths(Val(policy))
        # Construct hidden layers
        layers = Any[]
        for i in 1:length(layer_sizes) - 2
            push!(
                layers,
                Dense(
                    layer_sizes[i],
                    layer_sizes[i+1],
                    getproperty(NNlib, Symbol(act[i]));
                    init_weight=init,
                ),
            )
        end

        # Construct the final weight layer
        final_weight_layer = Dense(
            layer_sizes[end-1],
            layer_sizes[end],
            final_activation isa Tuple ? final_activation[i] : final_activation,
            init_weight=last_layer_init,
        )
        push!(layers, final_weight_layer)

        # Construct the chain of operations and add it to the network
        push!(paths, Chain(layers...))
    end

    return if n_paths(Val(policy)) > 1 Parallel(
        nothing,
        paths...
    )
    else
        paths[1]
    end
end

function final_act(
    ::Union{
        Val{:ArctanhNormalPolicy},
        Val{:LaplacePolicy},
        Val{:LogisticPolicy},
        Val{:LogitNormalPolicy},
        Val{:NormalPolicy},
        Val{:TruncatedLaplacePolicy},
        Val{:TruncatedNormalPolicy},
    },
    env
)
    # TODO: Probably better to use exp(LOG_STD_THRESHOLD * tanh(x)) or softplus(x) *
    # STD_THRESHOLD
    return let LOG_STD_THRESHOLD = LOG_STD_THRESHOLD
        identity,  x -> exp(clamp(x, -LOG_STD_THRESHOLD, LOG_STD_THRESHOLD))
    end
end

function final_act(
    ::Union{
        Val{:LaplacePolicy},
        Val{:LogisticPolicy},
        Val{:NormalPolicy},
    },
    env
)
    # TODO: Probably better to use exp(LOG_STD_THRESHOLD * tanh(x)) or softplus(x) *
    # STD_THRESHOLD
    action_scale = ActorCritic.high(action_space(env))[1]
    @assert (
        ActorCritic.high(action_space(env))[1] ==
        abs(ActorCritic.low(action_space(env))[1])
    )
    return let LOG_STD_THRESHOLD = LOG_STD_THRESHOLD
        (
            x -> action_scale .* tanh(x),
            x -> exp(clamp(x, -LOG_STD_THRESHOLD, LOG_STD_THRESHOLD)),
        )
    end
end

function final_act(::Union{Val{:BetaPolicy}, Val{:KumaraswamyPolicy}}, env)
    x -> softplus(x) .+ 1f-1
end

final_act(::Val{:GammaPolicy}, env) = (exp, exp)
final_act(::Val{:SoftmaxPolicy}, env) = identity

const TwoParamPolicy = Union{
    Val{:NormalPolicy},
    Val{:LaplacePolicy},
    Val{:GammaPolicy},
    Val{:LogisticPolicy},
    Val{:TruncatedNormalPolicy},
    Val{:TruncatedLaplacePolicy},
    Val{:BetaPolicy},
    Val{:KumaraswamyPolicy},
    Val{:LogitNormalPolicy},
    Val{:ArctanhNormalPolicy},
}

n_paths(::TwoParamPolicy) = 2
n_paths(::Val{:SoftmaxPolicy}) = 1
####################################################################

####################################################################
# Critic construction
####################################################################
function construct_critic_approximator(config::Dict{String,Any}, env, rng)
    @param_from n config
    @param_from approx config
    @param_from type config

    return  construct_critic_approximator(Val(Symbol(type)), approx, n, env, rng)
end

get_critic_outputs(::Val{:V}, env) = 1
function get_critic_outputs(::Val{:Q}, env)
    return if ActorCritic.continuous(ActorCritic.action_space(env))
        1
    else
        ActorCritic.action_space(env).n[1]
    end
end

get_critic_inputs(::Val{:V}, env) = size(ActorCritic.observation_space(env))[1]
function get_critic_inputs(::Val{:Q}, env)
    return if ActorCritic.continuous(ActorCritic.action_space(env))
        env_in = size(ActorCritic.observation_space(env))[1]
        action_size = size(ActorCritic.action_space(env))[1]
        env_in + action_size
    else
        size(ActorCritic.observation_space(env))[1]
    end
end

function construct_critic_approximator(type_symbol, config, n, env, rng)
    @param_from type config
    approx_type = Symbol(type)

    in_size = get_critic_inputs(type_symbol, env)
    out_size = get_critic_outputs(type_symbol, env)
    return construct_critic_approximator(
        Val(approx_type), type_symbol, in_size, out_size, config, n, env, rng,
    )
end

function construct_critic_approximator(
    ::Val{:Tabular}, ::Val{:Q}, in_size, out_size, config, n, env, rng,
)
    @assert n == 1

    n_actions = action_space(env).n[1]
    msg = "expected output to equal number of actions $n_actions but got $out_size"
    @assert out_size == n_actions msg
    n_states = observation_space(env).n[1]
    msg = "expected 1 input but got $in_size"
    @assert in_size == 1 msg

    @param_from init config
    init = if occursin(".", init)
        init = split(init, ".")
        getproperty(eval(Symbol(init[1])), Symbol(init[2]))
    else
        eval(Symbol(init))
    end

    return Tabular(n_states, n_actions; init)
end


function construct_critic_approximator(
    ::Val{:Lux}, ::Val{:V}, in_size, out_size, config, n, env, rng,
)
    @param_from hidden config
    @param_from act config
    @assert length(hidden) == length(act)

    @param_from init config
    init = getproperty(Lux, Symbol(init))

    paths = []
    for i in 1:n
        # Construct hidden layers
        layers = Any[
            Dense(
                in_size,
                hidden[1],
                getproperty(NNlib, Symbol(act[1]));
                init_weight=init,
            ),
        ]
        for i in 1:length(hidden) - 1
            push!(
                layers,
                Dense(
                    hidden[i], hidden[i+1], getproperty(NNlib, Symbol(act[i]));
                    init_weight=init,
                ),
            )
        end

        # Construct the final weight layer
        final_weight_layer = Dense(hidden[end], out_size, init_weight=init)
        push!(layers, final_weight_layer)

        # Construct the chain of operations and add it to the network
        push!(paths, Chain(layers...))
    end

    if n > 1
        return Chain(
            Parallel(
                (args...) -> cat(args...; dims=ndims(args[1]) + 1),
                paths...
            ),
        )
    else
        return paths[1]
    end
end

function construct_critic_approximator(
    ::Val{:Lux}, ::Val{:Q}, in_size, out_size, config, n, env, rng,
)
    @param_from hidden config
    @param_from act config
    @assert length(hidden) == length(act)

    @param_from init config
    init = getproperty(Lux, Symbol(init))

    paths = []
    for i in 1:n
        # Construct hidden layers
        layers = Any[
            Dense(
                in_size,
                hidden[1],
                getproperty(NNlib, Symbol(act[1]));
                init_weight=init,
            ),
        ]
        for i in 1:length(hidden) - 1
            push!(
                layers,
                Dense(
                    hidden[i], hidden[i+1], getproperty(NNlib, Symbol(act[i])); init_weight=init,
                ),
            )
        end

        # Construct the final weight layer
        final_weight_layer = Dense(hidden[end], out_size, init_weight=init)
        push!(layers, final_weight_layer)

        # Construct the chain of operations and add it to the network
        push!(paths, Chain(layers...))
    end

    if ActorCritic.continuous(ActorCritic.action_space(env))
        if n > 1
            return Chain(
                ActorCritic.Concat(1),
                Parallel(
                    (args...) -> cat(args...; dims=ndims(args[1]) + 1),
                    paths...
                ),
            )
        else
            return Chain(ActorCritic.Concat(1), paths[1])
        end
    else
        if n > 1
            return Chain(
                Parallel(
                    (args...) -> cat(args...; dims=ndims(args[1]) + 1),
                    paths...
                ),
            )
        else
            return paths[1]
        end
    end
end
####################################################################

end
