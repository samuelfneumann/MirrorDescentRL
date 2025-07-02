"""
    DiscreteProximalMDMPO(n, ρ, ρ̃)

`DiscreteProximalMDMPO` uses a functional proximal mirror descent MPO update with discrete
actions. Crucially, the policy parameterization must ensure valid policies (i.e.
distributions must sum to 1). This ensures that no projection operator is needed to project
distributions back to a valid distribution space. See `SimplexProximalMDMPO` for an
implementation that works on the simplex, which needs a projection operation after making
policy updates.

The functional mirror descent update is applied on the policy distribution itself,
using a negative entropy mirror map.

When using softmax policies, the functional mirror descent update can be applied either on
the policy distribution itself or on the softmax logits using a negative entropy mirror map
or a log-sum-exp mirror map respectively.

Uses Mini-Batch style updates.
"""
struct DiscreteProximalMDMPO <: AbstractActorUpdate
    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature
    _kl_policy_coeff::Float32

    _inv_λ::Float32     # Inverse stepsize for mirror descent (functional) update
    _num_md_updates::Int

    _forward_direction::Bool
    _use_baseline::Bool

    function DiscreteProximalMDMPO(
        τ::Real, kl_policy_coeff::Real, md_λ::AbstractFloat, num_md_updates::Int,
        use_baseline::Bool, forward_direction::Bool,
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"
        @assert (kl_policy_coeff > 0) "expected kl_policy_coeff >= 0"

        return new(
            τ, kl_policy_coeff, inv(md_λ), num_md_updates, forward_direction, use_baseline,
        )
    end
end

function DiscreteProximalMDMPO(
    τ::Real, kl_policy_coeff::Real, md_λ::AbstractFloat, num_md_updates::Int;
    use_baseline::Bool, forward_direction::Bool,
)
    return DiscreteProximalMDMPO(
        τ, kl_policy_coeff, md_λ, num_md_updates, use_baseline, forward_direction,
    )
end

function setup(
    up::DiscreteProximalMDMPO,
    env::AbstractEnvironment,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{DiscreteProximalMDMPO}
    # Setup gradient cache
    return UpdateState(
        up,
        optim,
        (
            optim = Optimisers.setup(optim, π_θ),
            # Previous policy parameters for the KL update
            θ_t = π_θ,    # These are immutable
            state_t = π_st,  # These are immutable
            current_update = 1,
        )
    )
end

function update(
    st::UpdateState{DiscreteProximalMDMPO},
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    up = st._update

    # Frozen current policy parameters, must stay fixed during the MD update and only update
    # every up._num_md_updates
    θ_t = st._state.θ_t
    # State of the current policy, which will change during the MD update
    state_t = st._state.state_t

    ∇π, π_st, qf_st, st_t = if !up._forward_direction
        _rkl_gradient(
            up, π, π_f, π_θ, π_st, θ_t, state_t, qf, qf_f, qf_θ, qf_st, states,
        )
    else
        _fkl_gradient(
            up, π, π_f, π_θ, π_st, θ_t, state_t, qf, qf_f, qf_θ, qf_st, states,
        )
    end

    optim_state = st._state.optim
    optim_state, π_θ = Optimisers.update(optim_state, π_θ, only(∇π))

    next_update = mod(st._state.current_update, up._num_md_updates) + 1
    return UpdateState(
        st._update,
        st._optim,
        (
            optim = optim_state,
            θ_t = next_update == 1 ? π_θ : θ_t,
            state_t = next_update == 1 ? π_st : state_t,
            current_update = next_update,
        ),
    ), π_θ, π_st, qf_st
end

function _rkl_gradient(
    up::DiscreteProximalMDMPO,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    θ_t,
    state_t,
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    # Compute advantages
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q
    @assert ndims(adv) == 2

    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, states)
    π_t = exp.(lnπ_t)

    # Compute KL policy in πₜ
    π_KL_logits = adv ./ up._kl_policy_coeff
    π_KL_logits .-= maximum(π_KL_logits; dims=1)
    π_KL_logits_exp = exp.(π_KL_logits)
    π_KL_numerator = π_KL_logits_exp .* π_t
    # π_KL = π_KL_numerator ./ sum(π_KL_numerator; dims=1)
    lnπ_KL = π_KL_logits .+ lnπ_t .- log.(sum(π_KL_numerator; dims=1))

    scale1 = exp.(lnπ_KL .- lnπ_t) .- ((up._τ - up._inv_λ) .* lnπ_t)

    ∇π_θ = gradient(π_θ) do π_θ
        # Compute the gradient ∇J = 𝔼_{I*}[ln(π)]
        lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st, states)
        π_θ = exp.(lnπ_θ)

        scale2 = up._inv_λ .* ChainRulesCore.ignore_derivatives(lnπ_θ)

        loss = -sum((scale1 - scale2) .* π_θ; dims=1)
        gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st, state_t
end

 function _fkl_gradient(
     up::DiscreteProximalMDMPO,
     π::SoftmaxPolicy,
     π_f,    # actor policy model
     π_θ,    # actor policy model parameters
     π_st,   # actor policy model state
     θ_t,
     state_t,
     qf::DiscreteQ,
     qf_f,
     qf_θ,
     qf_st,
     states::AbstractArray, # Must be >= 2D
 )
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q
    batch_size = size(states)[end]

    lnπ_t, st_t = logprob(π, π_f, θ_t, state_t, states)
    π_t = exp.(lnπ_t)
    @tullio entropy_t[i] := -π_t[j, i] * lnπ_t[j, i]

    # Compute KL policy in πₜ
    π_KL_logits = adv ./ up._kl_policy_coeff
    π_KL_logits .-= maximum(π_KL_logits; dims=1)
    π_KL_logits_exp = exp.(π_KL_logits)
    π_KL_numerator = π_KL_logits_exp .* π_t
    π_KL = π_KL_numerator ./ sum(π_KL_numerator; dims=1)

    ς = 1 .+ up._τ .* (lnπ_t .+ reshape(entropy_t, 1, :))

    ∇π_θ = gradient(π_θ) do π_θ
        lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st, states)

        term1 = sum(π_KL .* lnπ_θ; dims=1)

        expectation_arg = (ς .- up._inv_λ) .* lnπ_θ
        term2 = sum(π_t .* expectation_arg; dims=1)

        loss = term2 - term1
        gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st, st_t
end

function _rkl_gradient(
    up::DiscreteProximalMDMPO,
    π::SoftmaxPolicy,
    π_f::Tabular,   # actor policy model
    π_θ,            # actor policy model parameters
    π_st,           # actor policy model state
    θ_t,
    state_t,
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray{Int}, # Must be >= 2D
)
    # Compute advantages
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q
    @assert ndims(adv) == 2
    batch_size = size(adv, 2)

    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, states)
    π_t = exp.(lnπ_t)

    # Compute KL policy in πₜ
    π_KL_logits = adv ./ up._kl_policy_coeff
    π_KL_logits .-= maximum(π_KL_logits; dims=1)
    π_KL_logits_exp = exp.(π_KL_logits)
    π_KL_numerator = π_KL_logits_exp .* π_t
    # π_KL = π_KL_numerator ./ sum(π_KL_numerator; dims=1)
    lnπ_KL = π_KL_logits .+ lnπ_t .- log.(sum(π_KL_numerator))

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st)
    gs = zero(Float32, π_f)
    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]

            lr_term = exp.(lnπ_KL .- lnπ_t)
            entropy_t_term = (up._τ - up._inv_λ) .* lnπ_t[:, i]
            entropy_term = up._inv_λ .* lnπ_θ[:, s_t]

            scale = lr_term .- entropy_t_term .- entropy_term

            ∇π_θ = _∇_softmax_tabular(π_θ.layer_1, s_t; sum_over_actions=false)
            ∇π_θ .*= reshape(scale, 1, :)
            ∇π_θ_term = dropdims(sum(∇π_θ; dims=2); dims=2)

            g_i[:, s_t] .-= (∇π_θ_term ./ batch_size)
            end
        g_i
    end

    return (gs,), π_st, qf_st, state_t
end

function _fkl_gradient(
    up::DiscreteProximalMDMPO,
    π::SoftmaxPolicy,
    π_f::Tabular,   # actor policy model
    π_θ,            # actor policy model parameters
    π_st,           # actor policy model state
    θ_t,
    state_t,
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray{Int}, # Must be >= 2D
)
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q
    @assert ndims(adv) == 2
    batch_size = size(adv, 2)

    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, states)
    π_t = exp.(lnπ_t)

    entropy_t = sum(π_t .* lnπ_t; dims=1)
    ς = 1 .+ up._τ .* (lnπ_t .+ entropy_t)

    # Compute KL policy in πₜ
    π_KL_logits = adv ./ up._kl_policy_coeff
    π_KL_logits .-= maximum(π_KL_logits; dims=1)
    π_KL_logits_exp = exp.(π_KL_logits)
    π_KL_numerator = π_KL_logits_exp .* π_t
    π_KL = π_KL_numerator ./ sum(π_KL_numerator; dims=1)

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = spzeros(Float32, π_f)
    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]

            ∇lnπ_θ = _∇ln_softmax_tabular(π_θ.layer_1, s_t; sum_over_actions=false)

            term1 = dropdims(sum(π_KL[:, i]' .* ∇lnπ_θ; dims=2); dims=2)

            expectation_arg = (ς[:, i] .- up._inv_λ)' .* ∇lnπ_θ
            term2 = dropdims(sum(π_t[:, i]' .* expectation_arg; dims=2); dims=2)

            g_i[:, s_t] .-= ((term1 .- term2) ./ batch_size)
            end
        g_i
    end

    return (gs,), π_st, qf_st, state_t
end
