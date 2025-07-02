module SimplexExperimentUtils

using Random
using ..ActorCritic
import Reproduce: @param_from

function construct_env(config, seed)
    @param_from type config
    type = getproperty(ActorCritic, Symbol(type))

    @param_from kwargs config
    ks = keys(kwargs)
    vs = values(kwargs)
    keywords = [Symbol(k) => v for (k, v) in zip(Symbol.(ks), vs)]

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    env = type(rng; keywords...)

    cutoff_episodes = "step_limit" in keys(config)
    return if cutoff_episodes
        @param_from step_limit config
        StepLimit(env, step_limit)
    else
        env
    end
end

function construct_tabular_approximator(config, env, rng)
    @param_from approx config
    @param_from type approx

    # Simplex experiment only supports tabular parameterizations
    @assert getproperty(ActorCritic, Symbol(type)) isa Type{Tabular}

    @param_from init approx
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

const construct_policy_approximator = construct_tabular_approximator
const construct_critic_approximator = construct_tabular_approximator

end
