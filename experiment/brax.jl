module BraxExperiment

using ActorCritic
using Lux
using CUDA
using LuxCUDA
using Logging
using Optimisers
using ProgressMeter
using Random
using Reproduce
using Dates

import ActorCritic: ExpUtils
import Dates: now
import Reproduce: @param_from

import ChoosyDataLoggers
import ChoosyDataLoggers: @data
ChoosyDataLoggers.@init
function __init__()
    ChoosyDataLoggers.@register
end

const BU = ActorCritic.BraxExperimentUtils
const CHECKPOINT_EVERY_DEFAULT = 250000

include("./checkpoint.jl")

Reproduce.@generate_config_funcs begin
    "checkpoint_every" => 10000 # Steps
    "seed" => 6
    "steps" => 1000000
    "env" => {
        "type" => "halfcheetah",
        "step_limit" => 1000,
        "kwargs" => Dict(
            "backend" => "generalized",
            # "garbage_collect_every" => 100000,
            # "continuous" => true,
            # "sparse_rewards" => false,
            # "trig_features" => false,
            "γ" => 0.99f0,
            "garbage_collect_every" => 25000,
        )
    }

    "agent" => {
        "device" => "cpu"
        "forward_direction" => true
        "λ" => 1f0
        "τ" => 0f0
        "num_md_updates" => 10
        "type" => "BatchQAgent"
        "batch_size" => 32
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
                     "hidden" => [64, 64],
                ),
            ),
            "optim" => Dict(
                "type" => "Adam",
                "η_scale" => 1f-1,
            ),
            "update" => Dict{String,Any}(
                "type" => "ContinuousCCEM",
                "n" => 10,
                "ρs" => [0.1, 0.2],
                "num_entropy_samples" => 1,
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

    actor_update_iden = get_actor_update_iden(config)
    critic_update_iden = get_critic_update_iden(config)
    env_iden = get_env_iden(config)

    # Create the checkpoint directory if it doesn't exist
    checkpoint_dir = "./checkpoint/brax/$env_iden/"*
        "$(actor_update_iden)_$(critic_update_iden)"
    mkpath(checkpoint_dir)

    # TODO: the hash for the checkpoint file should **not** depend on any details about the
    # checkpointer. Right now, since the checkpoint_every flag is included in configuration
    # files, its value alters the checkpoint file name. But, we don't want this! If we
    # decide to checkpoint less often, then we should still use an existing checkpoint file
    # if available!! Two options:
    #
    #   1. Checkpointing freq is done at set intervals which the user cannot change
    #   2. Checkpointing freq is controlled by an environment variable
    #   3. Checkpointing freq is determined by the config file, but the checkpoint file hash
    #      ignores any checkpoint-related stuff in in the config file when generating the
    #      hash. The BIG PROBLEM with this is that Reproduce will still change the hash of
    #      the experiment based on the checkpointing freq (etc.), so I think this idea is
    #      very bad!
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

        # Delete unneeded checkpoint file
        try
            if checkpoint_every < steps && isfile(checkpointfile(checkpointer))
                @info "Removing checkpoint file"
                rm(checkpointfile(checkpointer))
            end
        catch e
            @info "Could not remove checkpoint file: $e"
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
    # TODO: offline eval
    # TODO: offline eval
    # TODO: offline eval
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

########################################
# Agents
########################################
function construct_agent(agent_config, env, seed)
    @param_from type agent_config
    agent_type = getproperty(ActorCritic, Symbol(type))

    construct_agent(agent_type, agent_config, env, seed)
end

function scale_optim_eta(actor_optim_config, critic_optim_config)
    if "η_scale" in keys(actor_optim_config)
        η_scale = actor_optim_config["η_scale"]
        η = critic_optim_config["η"]
        actor_optim_config["η"] = η_scale * η
    end

    return actor_optim_config
end

function construct_agent(::Type{BatchQAgent}, agent_config, env, seed)
    @info "Agent Config"
    display(agent_config)
    rng = Xoshiro(seed)

    # ####################################################################
    # Actor
    # ####################################################################
    @param_from actor agent_config

    # Get the actor policy and policy approximator
    @param_from policy actor
    policy_f = BU.construct_policy_approximator(policy, env, rng)
    policy = ActorCritic.construct_policy(policy, env)

    # Get the actor optimiser. If the key `η_scale` appears in the optimiser config, then
    # scale the value of key `η` in the critic config by the value of `η_scale` in the actor
    # config. I.e., in the actor optim config, we store the scale of the actor learning rate
    # relative to the critic learning rate, not the actual actor learning rate
    @param_from optim actor
    optim_config = scale_optim_eta(optim, agent_config["critic"]["optim"])
    policy_optim = ActorCritic.get_optimiser(optim_config)

    @param_from update actor

    # Adjust the entropy regularization coefficient. If it is in actor update config, then
    # we don't do anything. Otherwise, copy it from the agent config to the actor update
    # config
    if !("τ" in keys(update))
        if !("τ" in keys(agent_config))
            error("τ must be specified in the agent config if not in the update config")
        end

        # Trick to ensure actor and critic updates use the same τ
        @param_from τ agent_config
        update["τ"] = τ
    end

    # Similar as above, except for the KL terms
    if !("λ" in keys(update))
        if ("λ" in keys(agent_config))
            # Trick to ensure actor and critic updates use the same τ
            @param_from λ agent_config
            update["λ"] = λ
        end
    end
    if !("num_md_updates" in keys(update))
        if ("num_md_updates" in keys(agent_config))
            # Trick to ensure actor and critic updates use the same τ
            @param_from num_md_updates agent_config
            update["num_md_updates"] = num_md_updates
        end
    end

    # Get the actor update
    policy_update = ActorCritic.get_update(update)
    set_actor_update_ratio!(agent_config, policy_update)
    @show agent_config["actor_update_ratio"]

    ####################################################################
    # Critic
    ####################################################################
    @param_from critic agent_config

    # Get the critic value function and value function approximator
    @param_from value_fn critic
    q = ActorCritic.construct_critic(value_fn, env)
    q_f = BU.construct_critic_approximator(value_fn, env, rng)

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
                    error(
                        "τ must be specified in the agent config if not in the update " *
                        "config",
                    )
                end
                @param_from τ agent_config
                regularisers[i]["τ"] = τ
            end

            # Same as above, except for the KL coefficient
            if (
                    regularisers[i]["type"] == "KLBellmanRegulariser" &&
                    !("λ" in keys(regularisers[i]))
            )
                if !("λ" in keys(agent_config))
                    error(
                        "λ must be specified in the agent config if not in the update " *
                        "config",
                    )
                end
                @param_from λ agent_config
                regularisers[i]["λ"] = λ
            end
            if (
                    regularisers[i]["type"] == "KLBellmanRegulariser" &&
                    !("num_md_updates" in keys(regularisers[i]))
            )
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

    # Get the critic update
    q_update = ActorCritic.get_update(update)
    set_critic_update_ratio!(agent_config, q_update)
    @show agent_config["critic_update_ratio"]

    # Replay Buffer
    @param_from buffer agent_config
    buffer = ActorCritic.construct_buffer(buffer, env)
    @param_from steps_before_learning agent_config
    @param_from batch_size agent_config

    # Target nets
    @param_from target agent_config
    @param_from polyak target
    @param_from refresh_interval target

    @param_from actor_update_ratio agent_config
    @param_from critic_update_ratio agent_config

    @param_from device agent_config
    device = get_device(device)

    agent = BatchQAgent(
        seed, env, policy, policy_f, policy_optim, policy_update, actor_update_ratio, q,
        q_f, q_optim, q_update, critic_update_ratio, refresh_interval, polyak, buffer;
        batch_size=batch_size, steps_before_learning=steps_before_learning, device=device,
    )
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
    return BU.construct_env(env, seed)
end

####################################################################
# Two cases make sense to me:
#   1. The actor and critic update at the same rate, and they use the same `num_md_updates`
#   2. Let `num_md_updates_actor` be the number of mirror descent updates the proximal
#      actor makes. Let `num_md_updates_critic` be the number of KL-regularized updates the
#      critic makes. Then, the actor updates every step, and the critic updates
#      `num_md_updates_actor` times every `num_md_updates_actor` steps with
#      `num_md_updates_critic == 1`.
#
# Then the critic target either updates every `num_md_updates_critic` or on every critic
# update.
#
# We are going to use (1)
####################################################################
function set_actor_update_ratio!(
    agent_config, actor_update::ActorCritic.AbstractActorUpdate,
)
    if "actor_update_ratio" in keys(agent_config)
        if !check_update_ratio(agent_config["actor_update_ratio"])
            @warn "overwriting actor_update_ratio in agent config"
        end
    end

    agent_config["actor_update_ratio"] = (updates = 1, steps = 1)
    return nothing
end

function set_critic_update_ratio!(
    agent_config, critic_update::ActorCritic.AbstractCriticUpdate,
)
    if "critic_update_ratio" in keys(agent_config)
        if !check_update_ratio(agent_config["critic_update_ratio"])
            @warn "overwriting critic_update_ratio in agent config"
        end
    end

    agent_config["critic_update_ratio"] = (updates = 1, steps = 1)
    return nothing
end

function check_update_ratio(ratio::NamedTuple)::Bool
    return ratio.updates == 1 && ratio.steps == 1
end

function check_update_ratio(ratio)::Bool
    return ratio[1] == 1 && ratio[2] == 1
end

function check_update_ratio(ratio::Integer)::Bool
    return ratio == 1
end
####################################################################

end
