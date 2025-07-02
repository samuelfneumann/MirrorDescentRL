# MirrorDescentRL

Tabular, Linear, and Deep Actor-Critic algorithms implemented in Julia with
[Lux.jl](https://github.com/LuxDL/Lux.jl).

## Installation

First clone the repo and jump into a Julia project
```bash
git clone git@github.com:samuelfneumann/MirrorDescentRL.git
cd MirrorDescentRL
julia --project
```
then
```juliaREPL
julia> ]activate .
julia> ]instantiate
```
Finally, set up the PyCall.jl module following the instructions
[here](https://github.com/JuliaPy/PyCall.jl)

## Running the Code

The top-level module is called `ActorCritic`. To run an experiment, use the
standard setup from [Reproduce.jl](https://github.com/mkschleg/Reproduce.jl).

```julia
using Pkg
Pkg.activate(".")
using Reproduce

const CONFIG = "./config/template.toml"
const SAVE_PATH = "./results"

println("current working directory", pwd())

experiment = Reproduce.parse_experiment_from_config(CONFIG, SAVE_PATH)
pre_experiment(experiment)
ret = job(experiment)
post_experiment(experiment, ret)
```

where you should replace `CONFIG` and `SAVE_PATH` with the corresponding
configuration file and save path that you would like to use. 

Alternatively, for debugging, each experiment file contains an embedded
configuration dictionary, following Reproduce.jl's standard setup. You can run
an experiment with this embedded experiment file using

```juliaREPL
julia> include("./experiment/continuous_default.jl")
julia> using .DefaultContinuousExperiment
julia> DefaultContinuousExperiment.working_experiment()
```

A number of example configuration files based on the experiments from our paper
_Investigating the Utility of Mirror Descent in Off-policy Actor-Critic_ are
given in the `config/` directory. Overall, this codebase contains
implementations of many more actor/critic updates, policy parameterizations,
value function types, RL agents, etc. than we considered in our paper. We have
included our entire codebase here to help streamline the development,
implementation, and overall study of RL algorithms using Julia.

## Citing

If you use our codebase, please cite our conference paper:

```
@inproceedings{neumann2025investigating,
    title = {Investigating the Utility of Mirror Descent in Off-policy Actor-Critic},
    booktitle = {Reinforcement Learning Conferece},
    year = {2025},
    author = {Samuel Neumann and Jiamin He and Adam White and Martha White}
}
```
