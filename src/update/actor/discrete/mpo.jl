"""
    DiscreteMPO

DiscreteMPO implements an MPO-like update, without a KL penalty
"""
struct DiscreteMPO <: AbstractActorUpdate
    _kl_policy_coeff::Float32
    _τ::Float32

    _use_baseline::Bool
    _ε::Float32

    function DiscreteMPO(kl_policy_coeff::Real, τ::Real, use_baseline, ε::Real=0f0)
        @assert (kl_policy_coeff > 0) "expected kl_policy_coeff > 0"
        @assert (τ >= 0) "expected τ >= 0"
        @assert (ε >= 0) "expected ε >= 0"

        return new(kl_policy_coeff, τ, use_baseline, ε)
    end
end

function DiscreteMPO(kl_policy_coeff::Real, τ::Real; use_baseline, ε::Real=0f0)
    return DiscreteMPO(kl_policy_coeff, τ, use_baseline, ε)
end

function setup(
    up::DiscreteMPO,
    ::AbstractEnvironment,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG,
)::UpdateState{DiscreteMPO}
    return UpdateState(
        up,
        optim,
        (π_optim = Optimisers.setup(optim, π_θ),)
    )
end

function setup(
    up::DiscreteMPO,
    π::SimplexPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Union{Nothing,Optimisers.AbstractRule},
    _::AbstractRNG,
)
    error("cannot use DiscreteMPO with simplex policies")
end

function update(
    st::UpdateState{DiscreteMPO},
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

    ∇π_θ, π_st, qf_st = _gradient(
        up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states,
    )

    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    return UpdateState(
        st._update,
        st._optim,
        (π_optim = π_optim_state,),
    ), π_θ, π_st, qf_st
end

function _gradient(
    up::DiscreteMPO,
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
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q

    # Calculate the KL policy in π
    scale = adv ./ up._kl_policy_coeff
    scale = exp.(scale .- maximum(scale; dims=1))

    batch_size = size(states)[end]

    ∇π_θ = gradient(π_θ) do π_θ
        # Compute the gradient ∇J = 𝔼_{I*}[ln(π)]
        lnπ, π_st = logprob(π, π_f, π_θ, π_st, states)

        # Because we are in the discrete-action setting, we can directly compute π_{KL} and
        # use it rather than using values that are ∝ π_{KL}(a ∣ s)
        probs = ChainRulesCore.ignore_derivatives(exp.(lnπ))
        π_KL_numerator = scale .* probs .+ up._ε
        π_KL = ChainRulesCore.ignore_derivatives(
            π_KL_numerator ./ sum(π_KL_numerator; dims=1)
        )

        if up._τ > 0
            # Compute entropy regularization
            H, π_st = entropy(π, π_f, π_θ, π_st, states)
            -gpu_mean(sum(π_KL .* lnπ; dims=1)) - up._τ * gpu_mean(H)
        else
            -gpu_mean(sum(π_KL .* lnπ; dims=1))
        end
    end

    return ∇π_θ, π_st, qf_st
end

function _gradient(
    up::DiscreteMPO,
    π::SoftmaxPolicy,
    π_f::Tabular,   # actor policy model
    π_θ,            # actor policy model parameters
    π_st,           # actor policy model state
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractMatrix{Int},
)
    # Compute advantages
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st)
    adv = up._use_baseline ? q .- mean(q; dims=1) : q
    @assert ndims(adv) == 2

    # Compute KL policy in πₜ
    probs, π_st = prob(π, π_f, π_θ, π_st)
    π_KL_logits = adv ./ up._kl_policy_coeff
    π_KL_logits_exp = exp.(π_KL_logits .- maximum(π_KL_logits; dims=1))
    π_KL_numerator = π_KL_logits_exp .* probs .+ up._ε
    π_KL = π_KL_numerator ./ sum(π_KL_numerator; dims=1)

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = zero(Float32, π_f)
    batch_size = size(states, 2)
    gs = treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]
            g = -π_KL[:, s_t]' .* _∇ln_softmax_tabular(
                π_θ.layer_1, s_t; sum_over_actions=false,
            )
            g = sum(g; dims=2)

            if up._τ > zero(up._τ)
                g .-= (up._τ .* _∇entropy_softmax_tabular(π_θ.layer_1, s_t))
            end

            g_i[:, s_t] .+= (g ./ batch_size)
        end
        g_i
    end

    return (gs,), π_st, qf_st
end
