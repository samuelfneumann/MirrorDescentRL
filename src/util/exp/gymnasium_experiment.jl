module GymnasiumExperimentUtils

using Lux
using NNlib
using Random
using ..ActorCritic
import Reproduce: @param_from

const LOG_STD_THRESHOLD = log(1000f0)

function construct_env(config, seed)
    @param_from kwargs config
    ks = keys(kwargs)
    vs = values(kwargs)
    keywords = [Symbol(k) => v for (k, v) in zip(Symbol.(ks), vs)]

    env = if "name" in keys(config)
        @param_from name config

        version = if haskey(config, "version")
            @param_from version config
        else
            nothing
        end
        @show keywords
        ClipAction(GymnasiumEnv(name, version; seed=seed, unwrap=true, keywords...))
    else
        @param_from id config
        ClipAction(GymnasiumEnv(id; seed=seed, unwrap=true, keywords...))
    end

    cutoff_episodes = "step_limit" in keys(config)
    return if cutoff_episodes
        @param_from step_limit config
        StepLimit(env, step_limit)
    else
        env
    end
end

function construct_policy_approximator(config, env, rng)
    @param_from type config
    @param_from approx config
    @param_from type config

    return construct_policy_approximator(approx, Symbol(type), env, rng)
end

function construct_policy_approximator(config, policy::Symbol, env, rng)
    @param_from type config
    approx_type = Symbol(type)
    return construct_policy_approximator(Val(approx_type), policy, config, env, rng)
end

function construct_policy_approximator(::Val{:Lux}, policy::Symbol, config, env, rng)
    if ActorCritic.continuous(ActorCritic.action_space(env))
        in_size = size(ActorCritic.observation_space(env))[1]
        out_size = size(ActorCritic.action_space(env))[1]
    else
        in_size = size(ActorCritic.observation_space(env))[1]
        out_size = ActorCritic.action_space(env).n[1]
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

    paths = []
    for i in 1:n_paths(Val(policy))
        # Construct hidden layers
        layers = Any[
            Dense(in_size, hidden[1], getproperty(NNlib, Symbol(act[1])); init_weight=init),
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
        final_activation = final_act(Val(policy), env)
        final_weight_layer = Dense(
            hidden[end],
            out_size,
            final_activation isa Tuple ? final_activation[i] : final_activation,
            init_weight=init,
        )
        push!(layers, final_weight_layer)

        # Construct the chain of operations and add it to the network
        push!(paths, Chain(layers...))
    end

    return if n_paths(Val(policy)) > 1 Parallel(
        (args...) -> tuple(args...),
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
    return identity,  x -> exp(clamp(x, -LOG_STD_THRESHOLD, LOG_STD_THRESHOLD))
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
    return (
        x -> action_scale .* tanh(x),
        x -> exp(clamp(x, -LOG_STD_THRESHOLD, LOG_STD_THRESHOLD)),
    )
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

function construct_critic_approximator(config::Dict{String,Any}, env, rng)
    @param_from n config
    @param_from approx config
    @param_from type config

    return  construct_critic_approximator(Val(Symbol(type)), approx, n, env, rng)
end

function construct_critic_approximator(type, config, n, env, rng)
    error("construct_critic_approximator not implemented")
end

function construct_critic_approximator(::Val{:Q}, config, n, env, rng)
    @param_from type config
    approx_type = Symbol(type)

    if ActorCritic.continuous(ActorCritic.action_space(env))
        env_in = size(ActorCritic.observation_space(env))[1]
        action_size = size(ActorCritic.action_space(env))[1]
        in_size = env_in + action_size
        out_size = 1
    else
        env_in = size(ActorCritic.observation_space(env))[1]
        in_size = env_in
        out_size = ActorCritic.action_space(env).n[1]
    end
    return construct_critic_approximator(
        Val(approx_type), Val(:Q), in_size, out_size, config, n, env, rng,
    )
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

end
