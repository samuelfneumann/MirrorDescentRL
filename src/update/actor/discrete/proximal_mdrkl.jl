import ChoosyDataLoggers
import ChoosyDataLoggers: @data
# ChoosyDataLoggers.@init


"""
    DiscreteProximalMDRKL(reparameterised::Bool, baseline_actions::Int, τ::Real; num_samples=1)

Mirror Descent RKL

Uses Mini-Batch style updates, unlike MDRKL which uses a Dyna-style update
"""
struct DiscreteProximalMDRKL <: AbstractActorUpdate
    _temperature::Float32

    _inv_λ::Float32      # Inverse stepsize for mirror descent (functional) update
    _num_md_updates::Int

    _forward_direction::Bool

    function DiscreteProximalMDRKL(
        τ::Real, md_λ::AbstractFloat, num_md_updates::Int, forward_direction::Bool
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"

        return new(τ, inv(md_λ), num_md_updates, forward_direction)
    end
end

function setup(
    up::DiscreteProximalMDRKL,
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
    rng::AbstractRNG;
)::UpdateState{DiscreteProximalMDRKL}
    return UpdateState(
        up,
        optim,
        (
            optim = Optimisers.setup(optim, π_θ),
            θ_t = π_θ,    # These are immutable
            state_t = π_st,  # These are immutable
            current_update = 1,
        ),
    )
end

function update(
    st::UpdateState{DiscreteProximalMDRKL},
    π::AbstractDiscreteParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::DiscreteQ,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    states::AbstractArray, # Must be >= 2D
)
    up = st._update

    # Frozen current policy parameters, must stay fixed during the MD update and only change
    # every up._num_md_updates
    θ_t = st._state.θ_t
    # State of the current policy, which will change during the MD update
    state_t = st._state.state_t

    ∇π, π_st, qf_st, st_t = if !up._forward_direction
        _rkl_gradient(
            up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states, θ_t, state_t
        )
    else
        _fkl_gradient(
            up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states, θ_t, state_t
        )
    end

#     p_before, _ = ActorCritic.prob(
#         π, π_f, π_θ, π_st, states,
#     )

    π_optim_state = st._state.optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π))

#     p_after, _ = ActorCritic.prob(
#         π, π_f, π_θ, π_st, states,
#     )

#     log_p_before = log.(p_before)
#     log_p_before[isinf.(log_p_before)] .= 0f0
#     log_p_after = log.(p_after)
#     log_p_after[isinf.(log_p_after)] .= 0f0

#     @data exp norm=mean(mapslices(norm, p_before .- p_after; dims=1))
#     @data exp kl_before_after=mean(sum(p_before .* (log_p_before .- log_p_after); dims=1))
#     @data exp kl_after_before=mean(sum(p_after .* (log_p_after .- log_p_before); dims=1))

#     @show mean(mapslices(norm, p_before .- p_after; dims=1))
#     @show mean(sum(p_before .* (log_p_before .- log_p_after); dims=1))
#     @show mean(sum(p_after .* (log_p_after .- log_p_before); dims=1))

    next_update = mod(st._state.current_update, up._num_md_updates) + 1
    return UpdateState(
        st._update,
        st._optim,
        (
            optim = π_optim_state,
            θ_t = next_update == 1 ? π_θ : θ_t,
            state_t = next_update == 1 ? π_st : state_t,
            current_update = next_update,
        ),
    ), π_θ, π_st, qf_st
end

function _rkl_gradient(
    up::DiscreteProximalMDRKL,
    π::AbstractDiscreteParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    state_batch::AbstractArray,     # Must be >= 2D
    θ_t,                            # Previous policy parameters,
    state_t,                           # Previous policy state
)
    q_t, qf_st = predict(qf, qf_f, qf_θ, qf_st, state_batch)
    v_t = mean(q_t; dims=1)
    adv_t = q_t .- v_t

    lp_t, state_t = logprob(π, π_f, θ_t, state_t, state_batch)
    entropy_t_term = up._temperature * (1 + up._inv_λ) .* lp_t
    lp, π_t = logprob(π, π_f, π_θ, π_st, state_batch)
    entropy_term = up._temperature * up._inv_λ .* lp

    scale = adv_t .- entropy_t_term .+ entropy_term

    ∇π_θ = gradient(π_θ) do θ
        lnπ, π_st = logprob(π, π_f, θ, π_st, state_batch)
        prob = exp.(lnπ)

        loss = prob .* scale
        loss = sum(loss; dims=1)
        -gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st, state_t
end


function _fkl_gradient(
    up::DiscreteProximalMDRKL,
    π::SoftmaxPolicy,
    π_f,
    π_θ,
    π_st,
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    state_batch::AbstractArray,     # Must be >= 2D
    θ_t,                            # Previous policy parameters,
    state_t,                           # Previous policy state
)
    q_t, qf_st = predict(qf, qf_f, qf_θ, qf_st, state_batch)
    v_t = mean(q_t; dims=1)
    adv_t = q_t .- v_t

    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, state_batch)
    π_t = exp.(lnπ_t)
    entropy_t = sum(π_t .* lnπ_t; dims = 1)

    ς_t = adv_t .+ up._temperature .* (-lnπ_t .+ entropy_t)

    ∇π_θ = gradient(π_θ) do θ
        lnπ_θ, π_st = logprob(π, π_f, θ, π_st, state_batch)

        loss = (ς_t .+ up._inv_λ) .* lnπ_θ

        loss = π_t .* loss # Take expectation w.r.t. π_t
        loss = sum(loss; dims=1)

        -gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st, state_t
end

function _rkl_gradient(
    up::DiscreteProximalMDRKL,
    π::SoftmaxPolicy{F,N},
    π_f::Tabular,   # actor policy model
    π_θ,            # actor policy model parameters
    π_st,           # actor policy model state
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray{Int}, # Must be >= 2D
    θ_t,
    state_t,
) where {F,N}
    batch_size = size(states)[end]

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = spzeros(Float32, π_f)

    lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st)
    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t)
    π_t = exp.(lnπ_t)
    entropy_t = dropdims(sum(π_t .* lnπ_t; dims=1); dims=1)

    q_t, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    v_t = mean(q_t; dims=1)
    adv_t = q_t .- v_t

    gs = treemap(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]

            scale = (
                adv_t[:, i] .-
                up._temperature * (1 + up._inv_λ) .* lnπ_t[:, s_t] .+
                (up._temperature * up._inv_λ) .* lnπ_θ[:, s_t]
            )'
            ∇ = _∇_softmax_tabular(π_θ.layer_1, s_t; sum_over_actions=false)
            ∇ .*= scale
            ∇loss_θ = sum(∇; dims=2)

            g_i[:, s_t] .-= (∇loss_θ ./ batch_size)
        end
        g_i
    end
    gs = (gs,) # Keep consistent with Zygote

    return gs, π_st, qf_st, state_t
end

function _fkl_gradient(
    up::DiscreteProximalMDRKL,
    π::SoftmaxPolicy{F,N},
    π_f::Tabular,   # actor policy model
    π_θ,            # actor policy model parameters
    π_st,           # actor policy model state
    qf::DiscreteQ,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray{Int}, # Must be >= 2D
    θ_t,
    state_t,
) where {F,N}
    batch_size = size(states)[end]

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = spzeros(Float32, π_f)
    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t)
    π_t = exp.(lnπ_t)
    entropy_t = dropdims(sum(π_t .* lnπ_t; dims=1); dims=1)

    q_t, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    v_t = mean(q_t; dims=1)
    adv_t = q_t .- v_t

    gs = treemap(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]

            ς = adv_t[:, i] .+ up._temperature .* (-lnπ_t[:, s_t] .+ entropy_t[s_t])
            ∇ = _∇ln_softmax_tabular(π_θ.layer_1, s_t; sum_over_actions=false)
            ∇ .*= (ς' .* π_t[:, s_t])
            ∇loss_θ = dropdims(sum(∇; dims=2); dims=2)

            g_i[:, s_t] .-= (∇loss_θ ./ batch_size)
        end
        g_i
    end
    gs = (gs,) # Keep consistent with Zygote

    return gs, π_st, qf_st, state_t
end
