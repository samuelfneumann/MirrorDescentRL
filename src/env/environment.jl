iscallable(f) = !isempty(methods(f))

# Spaces
export Box, Discrete

include("spaces/abstract_space.jl")
include("spaces/box.jl")
include("spaces/discrete.jl")

# Environments

export
    AbstractEnvironment,
    MountainCar,
    Cartpole,
    Pendulum,
    Bimodal,
    TwoArmBandit,
    Gridworld,
    ContinuousGridworld,
    Acrobot,
    NoisyCliffWorld,
    CliffWorld,
    RewardSwitch,
    # BraxEnv,        # Re-export
    # GymnasiumEnv,   # Re-export
    AliasedState

export unwrap,
       unwrap_all,
       wrapped,
       start!,
       step!,
       stop!,
       render,
       γ,
       reward,
       isterminal,
       observation_space,
       action_space

include("environment/abstract_environment.jl")

function render end
function stop! end
function start! end
function step! end
function envstep! end

include("environment/cartpole.jl")
include("environment/mountain_car.jl")
include("environment/acrobot.jl")
include("environment/pendulum.jl")
include("environment/gridworld.jl")
include("environment/continuous_gridworld.jl")
include("environment/bimodal.jl")
include("environment/2_arm_bandit.jl")
include("environment/cliffworld.jl")
include("environment/noisy_cliffworld.jl")
include("environment/reward_switch.jl")
include("environment/aliased_state.jl")
# include("environment/brax.jl")
# include("environment/gymnasium.jl")

# using .Brax
# using .Gymnasium

# Environment wrappers
export AbstractWrapper, AbstractActionWrapper, AbstractRewardWrapper
export AbstractObservationWrapper
export
    StepLimit,
    ClipAction,
    CPUAction,
    GPUAction,
    Discretize,
    TransformReward,
    CastAction,
    CheckContainsAction,
    ObservationFunc,
    RewardNoise,
    CastObservation

include("environment/abstract_environment_wrapper.jl")
include("environment/wrapper/clip_action.jl")
include("environment/wrapper/cpu_action.jl")
include("environment/wrapper/gpu_action.jl")
include("environment/wrapper/cast_action.jl")
include("environment/wrapper/cast_observation.jl")
include("environment/wrapper/discretize.jl")
include("environment/wrapper/step_limit.jl")
include("environment/wrapper/transform_reward.jl")
include("environment/wrapper/check_contains_action.jl")
include("environment/wrapper/observation_func.jl")
include("environment/wrapper/reward_noise.jl")
