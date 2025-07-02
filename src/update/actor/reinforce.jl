struct REINFORCE <: AbstractPolicyGradientStyleUpdate
    # _reward_to_go::Bool # TODO
    _discount_weighted::Bool
end

function setup(
    up::REINFORCE,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG,
)
    return UpdateState(up, optim, (π_optim = Optimisers.setup(optim, π_θ),))
end

function update(
    st::UpdateState{REINFORCE},
    π::AbstractParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    s_t::AbstractArray, # Must be >= 2D
    a_t::AbstractArray,
    A_t::AbstractVector,
    γ_t::AbstractVector,
)
    up = st._update

    if up._discount_weighted
        error("discount weighted REINFORCE not implemented")
    else
        ∇π_θ = gradient(π_θ) do θ
            lnπ, π_st, = logprob(π, π_f, θ, π_st, s_t, a_t)
            -mean(lnπ .* A_t)
        end
    end

    # @data exp grad_norm=norm(only(∇π_θ) |> ComponentArray)
    _log_reinforce_data(π, π_f, π_θ, π_st, s_t)

    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    return UpdateState(
        st._update,
        st._optim,
        (π_optim = π_optim_state,),
    ), π_θ, π_st
end

function _log_reinforce_data(π::AbstractContinuousParameterisedPolicy, π_f, π_θ, π_st, s_t)
    params = get_params(π, π_f, π_θ, π_st, s_t)[1]

    policy = π(params...)
    _entropy_est = discrete_entropy.(policy)
    _entropy_est = mean(_entropy_est; dims=2)

    @data exp entropy_estimate=_entropy_est
end

function _log_reinforce_data(π::AbstractDiscreteParameterisedPolicy, π_f, π_θ, π_st, s_t)
    entropy, π_st = Distributions.entropy(π, π_f, π_θ, π_st, s_t)
    @data exp entropy_estimate=mean(entropy)
end

function discrete_entropy(d, n=1000)
    x = range(minimum(d), maximum(d), n)
    p = 0.0
    prev_cdf = 0.0
    H = 0.0
    for i in 2:n-1
        c = cdf(d, x[i])
        p = c - prev_cdf
        p = round(p, digits=10)
        H += p > 0 ? (p * log(p)) : 0.0
        prev_cdf = c
    end
    p = 1 - prev_cdf
    p = round(p, digits=10)
    H += p > 0 ? (p * log(p)) : 0.0
    return -H
end
