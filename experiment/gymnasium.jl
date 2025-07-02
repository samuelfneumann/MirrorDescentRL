# Similarly to classic_control.jl, we should alter the update ratios
module GymnasiumExperiment

using ActorCritic
using Lux
using CUDA
using LuxCUDA
using Logging
using Optimisers
using ProgressMeter
using Random
using Reproduce

import ActorCritic: ExpUtils
import Dates: now
import Reproduce: @param_from

import ChoosyDataLoggers
import ChoosyDataLoggers: @data
ChoosyDataLoggers.@init
function __init__()
    ChoosyDataLoggers.@register
end

const GU = ActorCritic.GymnasiumExperimentUtils
const DEU = ActorCritic.DefaultExperimentUtils
const CHECKPOINT_EVERY_DEFAULT = 100000

include("./checkpoint.jl")

Reproduce.@generate_config_funcs begin
    "seed" => 6
    "steps" => 1000000
    "checkpoint_every" => 10000
    "env" => {
        "id" => "TruckBackerUpper",
        # "version" => 4,
        "step_limit" => 300,
        "kwargs" => Dict(
            # "backend" => "generalized",
            "γ" => 0.99f0,
        )
    }

    "agent" => {
        "device" => "cpu"
        "forward_direction" => false
        "λ" => 1f0
        "τ" => 0f0
        "num_md_updates" => 10
        "type" => "BatchQAgent"
        "batch_size" => 256
        "steps_before_learning" => 1000
        "critic_update_ratio" => [1, 1]
        "actor_update_ratio" => (updates=1, steps=1)


        "actor" => Dict(
            "policy" => Dict(
                "type" => "ArctanhNormalPolicy",
                "approx" => Dict(
                     "type" => "Lux",
                     "init" => "glorot_uniform",
                     "act" =>  ["relu", "relu"],
                     "hidden" => [256, 256],
                ),
            ),
            "optim" => Dict(
                "type" => "Adam",
                "η_scale" => 1f-1,
            ),
            "update" => Dict{String,Any}(
                # "type" => "ContinuousProximalMDRKL",
                "type" => "ContinuousRKL",
                "baseline_actions" => 30,
                "reparam" => false,
                "num_samples" => 1,
            ),
        )

        "critic" => Dict(
            "value_fn" => Dict(
                "type" => "Q",
                "n" => 1,
                "approx" => Dict(
                     "type" => "Lux",
                     "init" => "glorot_uniform",
                     "act" =>  ["relu", "relu"],
                     "hidden" => [64, 64],
                ),
            ),
            "optim" => Dict(
                "type" => "Adam",
                "η" => 1f-2,
                "β" => [0.9, 0.999],
            ),
            "update" => Dict{String,Any}(
                "type" => "Sarsa",
                "regularisers" => [
                    Dict{String,Any}(
                        "type" => "EntropyBellmanRegulariser",
                    ),
                ]
            ),
        )
        "target" => Dict(
            "polyak" => 0.01f0,
            "refresh_interval" => 1,
        )
        "buffer" => Dict(
            "type" => "ExperienceReplay",
            "capacity" => 100000,
        )
    }
end

Reproduce.@generate_working_function

function main_experiment(config; progress=false, testing=false)
    @param_from steps config
    @param_from seed config
    checkpoint_every = get(config, "checkpoint_every", CHECKPOINT_EVERY_DEFAULT)

    # Create the checkpoint directory if it doesn't exist
    checkpoint_dir = "checkpoint/gymnasium"
    mkpath(checkpoint_dir)

    _HASH = hash(config)
    checkpoint_file = repr(_HASH) * ".jld2"
    checkpoint_md = CheckpointMetaData(checkpoint_dir, checkpoint_file)
    checkpoint = load_checkpoint(checkpoint_md)
    checkpointer = Checkpointer(checkpoint_every, checkpoint_md)

    Random.seed!(seed)

    Reproduce.experiment_wrapper(config; testing=testing) do config
        data, logger = if !checkpointed(checkpoint)
            extras = union(get(config, "log_extras", []), get(config, "save_extras", []))
            ExpUtils.construct_logger(
            steps=steps,
            extra_groups_and_names=extras,
        )
        else
            get_data(checkpoint, :logger_data), get_data(checkpoint, :logger)
        end

        with_logger(logger) do
            env, agent = if !checkpointed(checkpoint)
                @param_from agent config
                env = construct_env(config, seed)
                env, construct_agent(agent, env, seed)
            else
                @info "loading agent"
                env = construct_env(config, seed)
                env, get_data(checkpoint, :agent)
            end

            experiment_loop(
                agent, env, steps, checkpoint, checkpointer;
                logger=logger, logger_data=data, progress=progress, testing=testing,
            )
        end

        return save_results = ExpUtils.prep_save_results(
            data, get(config, "save_extras", []),
        )
    end

end

function experiment_loop(
    agent, env, num_steps, checkpoint::Checkpoint, checkpointer::Checkpointer; logger,
    logger_data, progress=false, testing=false,
)
    steps, eps, done, runtime = if checkpointed(checkpoint)
        steps = get_data(checkpoint, :steps)
        eps = get_data(checkpoint, :eps)
        done = get_data(checkpoint, :done)
        runtime = get_data(checkpoint, :runtime)
        @info "number of steps loaded from checkpoint: $steps"
        @info "number of episodes loaded from checkpoint: $eps"
        @info "total runtime up until checkpoint: $runtime"

        if done
            @info "loaded checkpoint from completed experiment, exiting..."
            return
        end

        steps, eps, done, runtime
    else
        0, 0, false, Millisecond(0)
    end

    if progress || true
        p = Progress(num_steps)
        ProgressMeter.update!(p, steps)
    end

    if !(env isa StepLimit)
        error("expected env to be a StepLimit but got $(typeof(env))")
    end

    start = now()
    while !done
        steps_to_run = minimum((env.steps_per_episode, num_steps - steps))
        ep_start = now()

        # Run another episode
        ep_reward, ep_steps = ActorCritic.run!(env, agent, steps_to_run) do (
            s_t, a_t, r_tp1, s_tp1, t, γ_tp1, agent_ret,
        )
            nothing
        end

        # Log stuff after the episode is over
        eps += 1
        if testing
            ep_time = now() - ep_start
            @show ep_reward
            @show ep_time
            @show ep_steps
        end

        steps += ep_steps
        done = steps >= num_steps

        # Log the total number of steps per episode
        @data exp ep_reward=ep_reward
        @data exp ep_steps=ep_steps

        if progress || true
            ProgressMeter.update!(p, steps)
        end

        if steps < num_steps
            write_checkpoint(
                checkpointer, steps;
                agent, env, steps, eps, logger, logger_data, done,
                runtime=(runtime + now() - start),
            )
        end

        if done
            @data exp runtime=(now()-start + runtime)
            return
        end
    end
end

########################################
# Configuration
########################################

function scale_optim_eta(actor_optim_config, critic_optim_config)
    if "η_scale" in keys(actor_optim_config)
        η_scale = actor_optim_config["η_scale"]
        η = critic_optim_config["η"]
        actor_optim_config["η"] = η_scale * η
    end

    return actor_optim_config
end

"""
    get_update_ratio(actor_update_type, agent_config)

Determines how many steps it takes for the actor to update `num_md_updates` times based on
the agent configuration `agent_config`. If the agent configuration does not have an actor
update ratio specified, then this function will return an update ratio such that the actor
makes `num_md_updates` every step.

After obtaining the actor update ratio, then this function adjusts the critic's update ratio
to make the same number of updates (`num_md_updates`) every `n` steps, where `n` is chosen
such that the critic only updates once the actor's proximal MD update has completed.

There is one special case. If the actor makes `num_md_updates` every
`num_md_updates` environment steps, then the critic will update once every environment step.

For example, if the `num_md_updates` and actor update ratio are set as in the table below,
then this function will return the following critic update ratio

|num_md_updates | # actor updates | every # steps || critic updates | every # steps |
|---------------|-----------------|---------------||----------------|---------------|
| 10            | 5               | 2             || 10             | 2             |
| 10            | 10              | 1             || 10             | 1             |
| 10            | 10              | 10            || 1              | 1             |
|---------------|-----------------|---------------||----------------|---------------|

The critic update ratio returned by this function can be overridden. If a critic update
ratio is present in the `agent_config`, then that critic update ratio is returned, rather
than the calculated one as described above.
"""
function get_update_ratio(::Type{<:ProximalMDUpdate}, agent_config)
    @param_from num_md_updates agent_config

    actor_update_ratio = if "actor_update_ratio" in keys(agent_config)
        @param_from actor_update_ratio agent_config
    elseif "actor_updates" in keys(agent_config)
        @param_from actor_updates agent_config
        @param_from actor_steps agent_config
        (updates=actor_updates, steps=actor_steps)
    else
        (updates=num_md_updates, steps=1)
    end
    actor_updates = actor_update_ratio[1]
    actor_steps = actor_update_ratio[2]

    @assert num_md_updates % actor_updates == 0

    critic_update_ratio = if "critic_update_ratio" in keys(agent_config)
        @param_from critic_update_ratio agent_config
    elseif "critic_updates" in keys(agent_config)
        @param_from critic_updates agent_config
        @param_from critic_steps agent_config
        (updates=critic_updates, steps=critic_steps)
    else
        steps_for_md_update = (num_md_updates ÷ actor_updates) * actor_steps

        if actor_updates == num_md_updates
            (num_md_updates ÷ steps_for_md_update, 1)
        else
            (num_md_updates, steps_for_md_update)
        end
    end

    return actor_update_ratio, critic_update_ratio
end

function get_update_ratio(::Type{<:AbstractActorUpdate}, agent_config)
    actor_update_ratio = if "actor_update_ratio" in keys(agent_config)
        @param_from actor_update_ratio agent_config
    elseif "actor_updates" in keys(agent_config)
        @param_from actor_updates agent_config
        @param_from actor_steps agent_config
        (updates=actor_updates, steps=actor_steps)
    elseif "update_ratio" in keys(agent_config)
        @param_from update_ratio agent_config
        (updates=update_ratio[1], steps=update_ratio[2])
    else
        @param_from updates agent_config
        @param_from steps agent_config
        (updates=updates, steps=steps)
    end

    critic_update_ratio = if "critic_update_ratio" in keys(agent_config)
        @param_from critic_update_ratio agent_config
    elseif "critic_updates" in keys(agent_config)
        @param_from critic_updates agent_config
        @param_from critic_steps agent_config
        (updates=critic_updates, steps=critic_steps)
    elseif "update_ratio" in keys(agent_config)
        @param_from update_ratio agent_config
        (updates=update_ratio[1], steps=update_ratio[2])
    else
        @param_from updates agent_config
        @param_from steps agent_config
        (updates=updates, steps=steps)
    end

    return actor_update_ratio, critic_update_ratio
end

####################
# Agent Construction
####################
function construct_agent(agent_config, env, seed)
    @param_from type agent_config
    agent_type = getproperty(ActorCritic, Symbol(type))

    construct_agent(agent_type, agent_config, env, seed)
end

function construct_agent(::Type{BatchQAgent}, agent_config, env, seed)
    @info "Using seed: $seed"
    @info "Agent Config"
    display(agent_config)
    rng = Xoshiro(seed)

    # ####################################################################
    # Actor
    # ####################################################################
    @param_from actor agent_config

    # Get the actor policy and policy approximator
    @param_from policy actor
    policy_f = DEU.construct_policy_approximator(policy, env, rng)
    policy = ActorCritic.construct_policy(policy, env)

    # Get the actor optimiser. If the key `η_scale` appears in the optimiser config, then
    # scale the value of key `η` in the critic config by the value of `η_scale` in the actor
    # config. I.e., in the actor optim config, we store the scale of the actor learning rate
    # relative to the critic learning rate, not the actual actor learning rate
    @param_from optim actor
    optim_config = scale_optim_eta(optim, agent_config["critic"]["optim"])
    policy_optim = ActorCritic.get_optimiser(optim_config)

    @param_from update actor
    @param_from type update
    actor_update_type = getproperty(ActorCritic, Symbol(type))
    policy_update = construct_actor_update(actor_update_type, agent_config, update)

    ####################################################################
    # Critic
    ####################################################################
    @param_from critic agent_config

    # Get the critic value function and value function approximator
    @param_from value_fn critic
    q = ActorCritic.construct_critic(value_fn, env)
    q_f = DEU.construct_critic_approximator(value_fn, env, rng)

    # Get the critic optimiser
    @param_from optim critic
    q_optim = ActorCritic.get_optimiser(optim)

    # Copy regularization coefficients from the agent config if they don't exists in the
    # regularizer configs
    @param_from update critic
    if "regularisers" in keys(update)
        @param_from regularisers update
        for i in 1:length(regularisers)
            # Adjust the entropy regularization coefficient. If it is in critic update
            # config, then we don't do anything. Otherwise, copy it from the agent config to
            # the critic update config
            if (
                    regularisers[i]["type"] == "EntropyBellmanRegulariser" &&
                    !("τ" in keys(regularisers[i]))
            )
                if !("τ" in keys(agent_config))
                    error("τ must be specified in the agent config if not in the update config")
                end
                @param_from τ agent_config
                regularisers[i]["τ"] = τ
            end

            if regularisers[i]["type"] == "KLBellmanRegulariser"
                # Same as above, except for the KL terms
                if !("λ" in keys(regularisers[i]))
                    if !("λ" in keys(agent_config))
                        error(
                            "λ must be specified in the agent config if not in the update " *
                            "config",
                        )
                    end
                    @param_from λ agent_config
                    regularisers[i]["λ"] = λ
                end

                if !("forward_direction" in keys(regularisers[i]))
                    if !("forward_direction" in keys(agent_config))
                        error(
                            "forward_direction must be specified in the agent config if not in the update " *
                            "config",
                        )
                    end
                    @param_from forward_direction agent_config
                    regularisers[i]["forward_direction"] = forward_direction
                end

                if !("num_md_updates" in keys(regularisers[i]))
                    if !("num_md_updates" in keys(agent_config))
                        error(
                            "num_md_updates must be specified in the agent config if not " *
                            "in the update config",
                        )
                    end
                    @param_from num_md_updates agent_config
                    regularisers[i]["num_md_updates"] = num_md_updates
                end
            end
        end
    end

    # Get the critic update
    q_update = ActorCritic.get_update(update)
    # set_critic_update_ratio!(agent_config, q_update)

    # Replay Buffer
    @param_from buffer agent_config
    buffer = ActorCritic.construct_buffer(buffer, env)
    @param_from steps_before_learning agent_config
    @param_from batch_size agent_config

    # Target nets
    @param_from target agent_config
    @param_from polyak target
    @param_from refresh_interval target

    # Get update ratios
    actor_update_ratio, critic_update_ratio = get_update_ratio(
        actor_update_type, agent_config,
    )
    @show actor_update_ratio, critic_update_ratio

    @param_from device agent_config
    device = get_device(device)

    agent = BatchQAgent(
        seed, env, policy, policy_f, policy_optim, policy_update, actor_update_ratio, q,
        q_f, q_optim, q_update, critic_update_ratio, refresh_interval, polyak, buffer;
        batch_size=batch_size, steps_before_learning=steps_before_learning, device=device,
    )

    return agent
end

########################## ######################### #########################
function construct_actor_update(
    type::Type{<:ClosedFormMirrorDescentUpdate}, agent_config, update_config,
)
    @param_from τ agent_config
    @param_from num_md_updates agent_config

    if "num_md_updates" in keys(agent_config) && agent_config["num_md_updates"] != 1
        msg = "num_md_updates must be 1 for closed-from mirror descent updates, " *
            "replacing num_md_updates in configuration"
        @warn msg
    end

    "τ" in keys(update_config) && @warn "replacing τ in actor update config"
    update_config["τ"] = τ

    @assert "forward_direction" ∉ keys(update_config)

    if "num_md_updates" in keys(update_config)
        @warn "replacing num_md_updates in actor update config"
    end
    update_config["num_md_updates"] = num_md_updates

    return ActorCritic.get_update(update_config)
end

function construct_actor_update(type::Type{<:ProximalMDUpdate}, agent_config, update_config)
    # Ensure the mirror descent stepsize is specified in the agent config
    @param_from λ agent_config
    @param_from τ agent_config
    @param_from num_md_updates agent_config
    @param_from forward_direction agent_config

    "λ" in keys(update_config) && @warn "replacing λ in actor update config"
    update_config["λ"] = λ
    "τ" in keys(update_config) && @warn "replacing τ in actor update config"
    update_config["τ"] = τ

    if "forward_direction" in keys(update_config)
        @warn "replacing forward_direction in update config"
    end
    update_config["forward_direction"] = forward_direction

    if "num_md_updates" in keys(update_config)
        @warn "replacing num_md_updates in actor update config"
    end
    update_config["num_md_updates"] = num_md_updates

    return ActorCritic.get_update(update_config)
end

function construct_actor_update(type, agent_config, update_config)
    @param_from τ agent_config

    "τ" in keys(update_config) && @warn "replacing τ in actor update config"
    update_config["τ"] = τ

    return ActorCritic.get_update(update_config)
end

get_device(iden::String) = lowercase(iden) == "cpu" ? cpu_device() : gpu_device()

function get_env_iden(config)
    @param_from env config
    @param_from type env
    return type
end

function get_actor_update_iden(config)
    @param_from agent config
    @param_from actor agent
    @param_from update actor
    @param_from type update
    return type
end

function get_critic_update_iden(config)
    @param_from agent config
    @param_from critic agent
    @param_from update critic
    @param_from type update
    return type
end
########################################

########################################
# Environments
########################################
function construct_env(config, seed)
    @param_from env config
    return GU.construct_env(env, seed)
end

end
