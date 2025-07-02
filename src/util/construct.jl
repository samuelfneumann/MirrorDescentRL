########################################
# Value Functions
########################################
function construct_critic(config, env; reduct=Base.minimum)
    @param_from type config

    return construct_critic(Val(Symbol(type)), env, reduct)
end

function construct_critic(type::Val, env, reduct)
    error("construct_critic not implemented")
end

function construct_critic(type::Val{:DiscreteQ}, env, reduct)
    return ActorCritic.DiscreteQ(reduct)
end

function construct_critic(type::Val{:Q}, env, reduct)
    if ActorCritic.continuous(ActorCritic.action_space(env))
        return ActorCritic.Q(reduct)
    else
        return ActorCritic.DiscreteQ(reduct)
    end
end

function construct_critic(type::Val{:V}, env, reduct)
    return ActorCritic.V(reduct)
end

########################################
# Policies
########################################
struct NoParamPolicyConfig end
struct OneParamPolicyConfig end
struct UnBoundedPolicyConfig end
struct MissingPolicyConfig end

get_policy_config(policy) = MissingPolicyConfig()

function get_policy_config(
    ::Union{
        Val{:GammaPolicy},
        Val{:LaplacePolicy},
        Val{:LogisticPolicy},
        Val{:NormalPolicy},
    },
)
    return UnBoundedPolicyConfig()
end

function get_policy_config(
    ::Union{
        Val{:TruncatedNormalPolicy},
        Val{:TruncatedLaplacePolicy},
        Val{:ArctanhNormalPolicy},
        Val{:LogitNormalPolicy},
        Val{:BetaPolicy},
        Val{:KumaraswamyPolicy},
        Val{:LogitPolicy},
        Val{:SoftmaxPolicy},
        Val{:SimplexPolicy},
    },
)
    return NoParamPolicyConfig()
end

function construct_policy(config, env)
    @param_from type config
    policy_type = Symbol(type)
    constr = getproperty(ActorCritic, policy_type)

    return construct_policy(constr, get_policy_config(Val(policy_type)), config, env)
end

function construct_policy(p, ::UnBoundedPolicyConfig, config, env)
    try
        @param_from clip_action config
        return p(env; clip_action=clip_action)
    catch e
        error("$p expects clip_action (bool): $e")
    end
end

function construct_policy(p, ::OneParamPolicyConfig, config, env)
    try
        @param_from ε config
        return p(ε)
    catch e
        error("$p expects ε (float): $e")
    end
end

function construct_policy(p, ::NoParamPolicyConfig, config, env)
    return p(env)
end

########################################
# Optimisers
########################################
"""
    get_optimiser

Return the optimiser given a config dictionary. The optimiser
name is found at key `"opt"`. The parameters also change based on the
optimiser.

- OneParamInit: `η::Float`
- TwoParamInit: `η::Float`, `ρ::Float`
- AdamParamInit: `η::Float`, `β::Vector` or `(β_m::Int, β_v::Int)`
"""
function get_optimiser(config::Dict)
    opt_iden = config["type"]
    get_optimiser(opt_iden, config)
end

function get_optimiser(opt_iden, config)
    if opt_iden isa Nothing || opt_iden == "nothing"
        # Sometimes we don't want an optimiser. This is signalled as Nothing
        return nothing
    end

    opt_type = getproperty(Optimisers, Symbol(opt_iden))
    return _init_optimiser(opt_type, config)
end

struct MissingParamInit end
struct OneParamInit end
struct TwoParamInit end
struct AdamParamInit end

param_init_style(opt_type) = MissingParamInit()

function param_init_style(
    ::Union{
        Type{Optimisers.Descent},
        Type{Optimisers.ADAGrad},
        Type{Optimisers.ADADelta},
    },
)
    return OneParamInit()
end

function param_init_style(
    ::Union{
        Type{Optimisers.RMSProp},
        Type{Optimisers.Momentum},
        Type{Optimisers.Nesterov},
    },
)
    return TwoParamInit()
end

function param_init_style(
    ::Union{
        Type{Optimisers.Adam},
        Type{Optimisers.RAdam},
        Type{Optimisers.NAdam},
        Type{Optimisers.AdaMax},
        Type{Optimisers.OAdam},
        Type{Optimisers.AMSGrad},
        Type{Optimisers.AdaBelief},
    },
)
    return AdamParamInit()
end

function _init_optimiser(opt_type, config::Dict)
    return _init_optimiser(param_init_style(opt_type), opt_type, config::Dict)
end

function _init_optimiser(::MissingParamInit, args...)
    error("optimiser initialization not found")
end

function _init_optimiser(::OneParamInit, opt_type, config::Dict)
    try
        @param_from η config
        return opt_type(η)
    catch e
        if e isa UndefVarError
            error("$(opt_type) needs: η (float): $e")
        else
            throw(e)
        end
    end
end

function _init_optimiser(::TwoParamInit, opt_type, config::Dict)
    try
        @param_from η config
        @param_from ρ config
        return opt_type(η, ρ)
    catch e
        if e isa UndefVarError
            error("$(opt_type) needs: η (float), and ρ (float): $e")
        else
            throw(e)
        end
    end
end

function _init_optimiser(::AdamParamInit, opt_type, config::Dict)
    try
        @param_from η config
        if "β" ∈ keys(config)
            @param_from β config
            return opt_type(η, tuple(β...))
        elseif "β_m" ∈ keys(config)
            @param_from β_m config
            @param_from β_v config
            β = (β_m, β_v)
            return opt_type(η, β)
        else
            return opt_type(η)
        end
    catch e
        if e isa UndefVarError
            error(
                "$(opt_type) needs: η (float), and β ((float, float)), or " *
                "(β_m, β_v) (float, float)): $e :: $config"
            )
        else
            throw(e)
        end
    end
end

########################################
# Actor Updates
########################################
function get_update(config)
    @param_from type config
    update = getproperty(ActorCritic, Symbol(type))
    return get_update(update, config)
end

function get_update(update_type::Type{REINFORCE}, config)
    try
        @param_from discount_weighted config
        return update_type(discount_weighted)
    catch e
        if e isa UndefVarError
            error("$(update_type) needs: discount_weighted (bool): $e")
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{PPO}, config)
    try
        @param_from clip_ratio config
        @param_from n_updates config
        @param_from spread_updates_across_env_steps config
        @param_from τ config
        return update_type(clip_ratio, n_updates, spread_updates_across_env_steps, τ)
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: clip_ratio (float), n_updates (int), " *
                "spread_updates_across_env_steps (bool), τ (float): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{NullActorUpdate}, config)
    return NullActorUpdate()
end

function get_update(update_type::Union{Type{SimplexCCEM},Type{SimplexMDCCEM}}, config)
    try
        @param_from τ config
        if "minp" in keys(config)
            @param_from minp config
            return update_type(τ; ensure_stochastic=true, minp)
        elseif "ensure_stochastic" in keys(config)
            @param_from ensure_stochastic config
            return update_type(τ; ensure_stochastic=ensure_stochastic)
        else
            return update_type(τ)
        end

    catch e
        if e isa UndefVarError
            error("$(update_type) needs: τ (float): $e")
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{DiscreteProximalMDMPO}, config)
    try
        @param_from forward_direction config
        @param_from τ config
        @param_from num_md_updates config
        @param_from λ config
        @param_from kl_policy_coeff config
        @param_from use_baseline config

        return update_type(
            τ, kl_policy_coeff, λ, num_md_updates; forward_direction, use_baseline,
        )
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: n (int), num_entropy_samples (int), τ (float), " *
                "num_md_updates (int), λ (float), kl_policy_coeff (float), " *
                "forward_direction (bool): $e"
            )
        else
            throw(e)
        end
    end
end


function get_update(update_type::Type{ContinuousProximalMDMPO}, config)
    try
        @param_from forward_direction config
        @param_from τ config
        @param_from num_md_updates config
        @param_from λ config
        @param_from kl_policy_coeff config
        @param_from baseline_actions config
        @param_from n config

        return update_type(
            kl_policy_coeff, τ, λ, num_md_updates; baseline_actions, n, forward_direction,
        )

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: n (int), τ (float), " *
                "num_md_updates (int), λ (float), kl_policy_coeff (float), " *
                "forward_direction (bool): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{ContinuousMPO}, config)
    try
        @param_from τ config
        @param_from n config
        @param_from kl_policy_coeff config
        @param_from baseline_actions config
        @param_from num_entropy_samples config
        return update_type(kl_policy_coeff, τ; baseline_actions, n, num_entropy_samples)
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: kl_policy_coeff (float), n (int), τ (float), " *
                "baseline_actions (int), and num_entropy_samples (int): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{DiscreteMPO}, config)
    try
        @param_from τ config
        @param_from kl_policy_coeff config
        @param_from use_baseline config
        if "ε" in keys(config)
            @param_from ε config
            return update_type(kl_policy_coeff, τ; use_baseline, ε)
        end
        return update_type(kl_policy_coeff, τ; use_baseline)
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: kl_policy_coeff (float), τ (float), " *
                "use_baseline (bool): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{ContinuousProximalMDCCEM}, config)
    if "num_samples" ∈ keys(config)
        @param_from num_samples config
        if num_samples isa Tuple || num_samples isa Vector
            π_num_samples, π̃_num_samples = num_samples
        elseif num_samples isa Integer
            π_num_samples = num_samples
            π̃_num_samples = num_samples
        else
            error("expected num_samples (int or (int, int))")
        end
    elseif "π_num_samples" ∈ keys(config)
        @param_from π_num_samples config
        @param_from π̃_num_samples config
    end

    if "τ̃" ∈ keys(config)
        @param_from τ̃ config
        @param_from τ config
    elseif "τs" in keys(config)
        try
            @param_from τs config
            τ, τ̃ = τs
        catch e
            if e isa BoundsError
                error("could not unpack τs: $e")
            else
                throw(e)
            end
        end
    else
        @param_from τ config
        τ̃ = τ
    end

    if "λ̃" ∈ keys(config)
        @param_from λ̃ config
        @param_from λ config
    elseif "λs" in keys(config)
        try
            @param_from λs config
            λ, λ̃ = λs
        catch e
            if e isa BoundsError
                error("could not unpack λs: $e")
            else
                throw(e)
            end
        end
    else
        @param_from λ config
        λ̃ = λ
    end

    clip = "clip" ∈ keys(config) ? config["clip"] : Inf

    @param_from num_md_updates config
    @param_from n config
    @param_from forward_direction config

    if "ρ̃" ∈ keys(config)
        @param_from ρ config
        @param_from ρ̃ config
    elseif "ρs" ∈ keys(config)
        try
            @param_from ρs config
            ρ, ρ̃ = ρs
        catch e
            if e isa BoundsError
                error("could not unpack ρs: $e")
            else
                throw(e)
            end
        end
    end

    try
        return update_type(
            n, ρ, ρ̃, τ, π_num_samples, τ̃, π̃_num_samples, λ, λ̃, num_md_updates,
            forward_direction, clip,
        )
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: n (int), ρs ((float, float)) or " *
                "(ρ, ρ̃) (float, float), τs ((float, float)) or (τ, τ̃) (float, float), " *
                "num_samples (int or (int, int)) or (π_num_samples, " *
                "π̃_num_samples (int, int), num_md_updates (int), " *
                "λs ((float, float)) or (λ, λ̃) (float, float), and forward_direction " *
                "(bool): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(
    update_type::Union{Type{SimplexMDMPO},Type{SimplexMPO}}, config,
)
    try
        @param_from τ config
        @param_from ensure_stochastic config
        @param_from kl_policy_coeff config
        @param_from use_baseline config

        if "minp" in keys(config)
            @param_from minp config
            return update_type(kl_policy_coeff, τ; ensure_stochastic, use_baseline, minp)
        end
        return update_type(kl_policy_coeff, τ; ensure_stochastic, use_baseline)

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: τ (float), ensure_stochastic (bool), " *
                "[minp (float)], kl_policy_coeff (float), and " *
                "use_baseline (bool): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(
    update_type::Type{SimplexProximalMDMPO},
    config,
)
    try
        @param_from τ config
        @param_from num_md_updates config
        @param_from λ config
        @param_from ensure_stochastic config
        @param_from kl_policy_coeff config
        @param_from use_baseline config

        if "minp" in keys(config)
            @param_from minp config
            return update_type(
                kl_policy_coeff, τ, num_md_updates, λ; ensure_stochastic, use_baseline,
                minp,
            )
        end
        return update_type(
            kl_policy_coeff, τ, num_md_updates, λ; ensure_stochastic, use_baseline,
        )

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: τ (float), num_md_updates (int), λ (float), " *
                "ensure_stochastic (bool), [minp (float)], use_baseline (bool), " *
                "and kl_policy_coeff (float): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{SimplexProximalMDCCEM}, config)
    try
        @param_from τ config
        @param_from num_md_updates config
        @param_from λ config
        @param_from ensure_stochastic config

        if "minp" in keys(config)
            @param_from minp config
            return update_type(τ, num_md_updates, λ; ensure_stochastic, minp)
        end
        return update_type(τ, num_md_updates, λ; ensure_stochastic)

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: τ (float), num_md_updates (int), λ (float), " *
                "ensure_stochastic (bool), minp (float): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Union{Type{OnPolicySimplexPG},Type{SimplexPG}}, config)
    try
        @param_from γ config
        return update_type(;γ)
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: γ (float): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{SimplexSPMD}, config)
    try
        @param_from adaptive_step_size config
        @param_from max_step_size config
        return update_type(;adaptive_step_size, max_step_size)
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: adaptive_step_size (bool) and max_step_size " *
                "(float): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{SimplexPMD}, config)
    try
        @param_from λ config
        return update_type(λ)
    catch e
        if e isa UndefVarError
            error("$(update_type) needs: λ (float): $e")
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{SimplexProximalMDRKL}, config)
    try
        @param_from τ config
        @param_from num_md_updates config
        @param_from λ config
        @param_from ensure_stochastic config
        @param_from use_baseline config

        if "minp" in keys(config)
            @param_from minp config
            return update_type(
                τ, num_md_updates, λ; ensure_stochastic, use_baseline, minp,
            )
        end
        return update_type(τ, num_md_updates, λ; ensure_stochastic, use_baseline)

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: τ (float), num_md_updates (int), λ (float), " *
                "ensure_stochastic (bool), minp (float), use_baseline (bool): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{DiscreteCCEM}, config)
    try
        @param_from τ config
        if "ε" in keys(config)
            @param_from ε config
            return update_type(τ; ε)
        end
        return update_type(τ)
    catch e
        e isa UndefVarError ? error("$(update_type) needs: τ (float): $e") : throw(e)
    end
end

function get_update(update_type::Type{ContinuousCCEM}, config)
    if "num_entropy_samples" ∈ keys(config)
        @param_from num_entropy_samples config
        if num_entropy_samples isa Tuple || num_entropy_samples isa Vector
            π_num_entropy_samples, π̃_num_entropy_samples = num_entropy_samples
        elseif num_entropy_samples isa Integer
            π_num_entropy_samples = num_entropy_samples
            π̃_num_entropy_samples = num_entropy_samples
        else
            error("expected num_entropy_samples (int or (int, int))")
        end
    elseif "π_num_entropy_samples" ∈ keys(config)
        @param_from π_num_entropy_samples config
        @param_from π̃_num_entropy_samples config
    end

    if "τ̃" ∈ keys(config)
        @param_from τ̃ config
        @param_from τ config
    elseif "τs" in keys(config)
        try
            @param_from τs config
            τ, τ̃ = τs
        catch e
            if e isa BoundsError
                error("could not unpack τs: $e")
            else
                throw(e)
            end
        end
    else
        @param_from τ config
        τ̃ = τ
    end

    @param_from n config

    if "ρ̃" ∈ keys(config)
        @param_from ρ config
        @param_from ρ̃ config
    elseif "ρs" ∈ keys(config)
        try
            @param_from ρs config
            ρ, ρ̃ = ρs
        catch e
            if e isa BoundsError
                error("could not unpack ρs: $e")
            else
                throw(e)
            end
        end
    end

    try
        return update_type(n, ρ, ρ̃, τ, π_num_entropy_samples, τ̃, π̃_num_entropy_samples)
    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: n (int), ρs ((float, float)) or " *
                "(ρ, ρ̃) (float, float)), τs ((float, float)) or (τ, τ̃) (float, float), " *
                "num_entropy_samples (int or (int, int)) or " *
                "(π_num_entropy_samples, π̃_num_entropy_samples) (int, int): $e"
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{DiscreteRKL}, config)
    try
        @param_from τ config
        @param_from use_baseline config

        return update_type(τ, use_baseline)
    catch e
        if e isa UndefVarError
            error("$(update_type) needs: τ (float), use_baseline (bool): $e")
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{ContinuousRKL}, config)
    try
        @param_from τ config
        @param_from reparam config
        @param_from baseline_actions config
        @param_from num_samples config

        return update_type(τ; reparam, baseline_actions, num_samples)

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: τ (float), reparam (bool), " *
                "baseline_actions(int) and num_samples (int): $e",
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Union{Type{SimplexRKL},Type{SimplexMDRKL}}, config)
    try
        @param_from τ config
        @param_from use_baseline config
        if "minp" in keys(config)
            @param_from minp config
            return update_type(τ; ensure_stochastic=true, minp, use_baseline)
        elseif "ensure_stochastic" in keys(config)
            @param_from ensure_stochastic config
            return update_type(τ; ensure_stochastic, use_baseline)
        else
            return update_type(τ; use_baseline)
        end

    catch e
        if e isa UndefVarError
            error("$(update_type) needs: τ (float), use_baseline (bool): $e")
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{ContinuousProximalMDRKL}, config)
    try
        @param_from forward_direction config
        @param_from τ config
        @param_from num_md_updates config
        @param_from λ config
        @param_from reparam config
        @param_from baseline_actions config
        @param_from num_samples config

        return update_type(
            τ, λ, num_md_updates; forward_direction, num_samples, reparam, baseline_actions
        )

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: τ (float), reparam (bool), " *
                "baseline_actions(int), num_samples (int), λ (float), " *
                "num_md_updates (int), forward_direction (bool): $e",
            )
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{ContinuousRKLKL}, config)
    try
        @param_from forward_direction config
        @param_from τ config
        # TODO: this should be renamed to num_kl_updates
        @param_from num_md_updates config
        @param_from λ config
        @param_from reparam config
        @param_from baseline_actions config
        @param_from num_samples config

        return update_type(
            τ, λ, num_md_updates; forward_direction, num_samples, reparam, baseline_actions
        )

    catch e
        if e isa UndefVarError
            error(
                "$(update_type) needs: τ (float), reparam (bool), " *
                "baseline_actions(int), num_samples (int), λ (float), " *
                "num_kl_updates (int), forward_direction (bool): $e",
            )
        else
            throw(e)
        end
    end
end

function get_update(
    update_type::Union{Type{DiscreteProximalMDRKL},Type{DiscreteProximalMDCCEM}}, config,
)
    @param_from forward_direction config
    @param_from τ config
    @param_from num_md_updates config
    @param_from λ config
    return update_type(τ, λ, num_md_updates, forward_direction)
end

# function get_update(update_type::Type{SoftmaxFMA}, config)
#     @show config
#     @param_from num_md_updates config
#     @param_from τ config
#     @param_from maxent config
#     @param_from λ config
#     @param_from force_q_t config

#     return update_type(τ, num_md_updates, λ; force_q_t, maxent)
# end

function get_update(update_type::Type{DiscreteFKL}, config)
    try
        @param_from τ config
        return update_type(τ)
    catch e
        e isa UndefVarError ? error("$(update_type) needs: τ (float): e") : throw(e)
    end
end

# function get_update(update_type::Type{FKL}, config)
#     try
#         @param_from τ config
#         num_samples = if "num_samples" in keys(config)
#             @param_from num_samples config
#         else
#             1
#         end

#         return update_type(τ; num_samples=num_samples)

#     catch e
#         if e isa UndefVarError
#             error("$(update_type) needs: τ (float): e")
#         else
#             throw(e)
#         end
#     end
# end
####################################################################

####################################################################
# Critic Updates
####################################################################
function get_regularizers(reg_configs)
    regs = []
    for config in reg_configs
        @param_from type config
        type = getproperty(ActorCritic, Symbol(type))
        reg = get_regularizer(type, config)
        if !(reg isa NullBellmanRegulariser)
            push!(regs, reg)
        end
    end
    return tuple(regs...)
end

function get_regularizer(type::Type{NullBellmanRegulariser}, config)
    return NullBellmanRegulariser()
end

function get_regularizer(type::Type{KLBellmanRegulariser}, config)
    try
        @param_from forward_direction config
        @param_from λ config
        @param_from num_md_updates config
        if λ == 0 || num_md_updates == 0
            return NullBellmanRegulariser()
        end
        if "num_samples" in keys(config)
            @param_from num_samples config
            return KLBellmanRegulariser(λ, num_md_updates, num_samples, forward_direction)
        else
            KLBellmanRegulariser(λ, num_md_updates, forward_direction)
        end

    catch e
        if e isa UndefVarError
            error(
                "$(type) needs: λ (float), num_md_updates (int), and " *
                "num_samples (int): $e",
            )
        else
            throw(e)
        end
    end
end

function get_regularizer(type::Type{EntropyBellmanRegulariser}, config)
    try
        @param_from τ config
        if τ == 0
            return NullBellmanRegulariser()
        end
        if "num_samples" in keys(config)
            @param_from num_samples config
            return EntropyBellmanRegulariser(τ, num_samples)
        else
            return EntropyBellmanRegulariser(τ)
        end

    catch e
        if e isa UndefVarError
            error("$(type) needs: τ (float) and num_samples (int): $e")
        else
            throw(e)
        end
    end
end

function get_update(update_type::Type{TD}, config)
    return update_type()
end

function get_update(update_type::Type{Sarsa}, config)
    try
        regs = if "regularisers" in keys(config)
            @param_from regularisers config
            get_regularizers(regularisers)
        else
            tuple()
        end

        return update_type(regs)

    catch e
        if e isa UndefVarError
            error("$(update_type) needs: reg (tuple of regularisers): $e")
        else
            throw(e)
        end
    end
end
####################################################################

########################################
# Replay Buffers
########################################
function construct_buffer(config, env)
    @param_from type config
    type = Symbol(type)
    return construct_buffer(Val(type), env, config)
end

function construct_buffer(::Val{:ExperienceReplay}, env, config)
    @param_from capacity config
    return ActorCritic.ExperienceReplay(env, capacity)
end

function construct_buffer(::Val{:VectorExperienceReplay}, env, config)
    @param_from capacity config
    return ActorCritic.VectorExperienceReplay(env, capacity)
end

function construct_buffer(::Val{:GAEBuffer}, env, config)
    @param_from capacity config
    @param_from γ config
    @param_from λ config
    return ActorCritic.GAEBuffer(env, capacity, γ, λ)
end
