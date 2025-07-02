"""
    DiscreteProximalMDCCEM

`DiscreteProximalMDCCEM` uses a functional proximal mirror descent CCEM update with discrete
actions. Crucially, the policy parameterization must ensure valid policies (i.e.
distributions must sum to 1). This ensures that no projection operator is needed to project
distributions back to a valid distribution space. See `SimplexProximalMDCCEM` for an
implementation that works on the simplex, which needs a projection operation after making
policy updates.

The functional mirror descent update is applied on the policy distribution itself,
using a negative entropy mirror map.

When using softmax policies, the functional mirror descent update can be applied either on
the policy distribution itself or on the softmax logits using a negative entropy mirror map
or a log-sum-exp mirror map respectively.

Uses Mini-Batch style updates.
"""
struct DiscreteProximalMDCCEM <: AbstractActorUpdate
    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature

    _inv_λ::Float32     # Inverse stepsize for mirror descent (functional) update
    _num_md_updates::Int

    _forward_direction::Bool

    function DiscreteProximalMDCCEM(
        τ::Real,  md_λ::Real, num_md_updates::Int, forward_direction::Bool,
    )
        @assert (num_md_updates > 1) "expected num_md_updates > 1"
        @assert (md_λ > 0f0) "expected functional stepsize md_λ > 0)"
        @assert (τ >= 0) "expected τ >= 0"

        return new(τ, inv(md_λ), num_md_updates, forward_direction)
    end
end

function DiscreteProximalMDCCEM(
    τ::Real,  md_λ::AbstractFloat, num_md_updates::Int; forward_direction::Bool,
)
    DiscreteProximalMDCCEM(τ, md_λ, num_md_updates, forward_direction)
end

function setup(
    up::DiscreteProximalMDCCEM,
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
)::UpdateState{DiscreteProximalMDCCEM}
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

function setup(
    up::DiscreteProximalMDCCEM,
    env::AbstractEnvironment,
    π::SimplexPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{DiscreteProximalMDCCEM}
    error("cannot use SimplexPolicy with DiscreteProximalMDCCEM")
end

function update(
    st::UpdateState{DiscreteProximalMDCCEM},
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
    up::DiscreteProximalMDCCEM,
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
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    batch_size = size(states)[end]

    # Find the indices of the actions of maximal value
    ind = mapslices(x -> sortperm(x; rev=true), q; dims=1)
    top_ind = [CartesianIndex(ind[1, j], j) for j in 1:batch_size]

    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t, states)
    best_lnπ_t = exp.(lnπ_t[top_ind])

    ∇π_θ = gradient(π_θ) do π_θ
        # Compute the gradient ∇J = 𝔼_{I*}[ln(π)]
        lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st, states)
        π_θ = exp.(lnπ_θ)
        best_lnπ_θ = lnπ_θ[top_ind]

        lr_term = exp.(best_lnπ_θ .- best_lnπ_t)
        kl_entropy_term = ChainRulesCore.ignore_derivatives(
            (up._τ - up._inv_λ) .* lnπ_t .+ up._inv_λ .* lnπ_θ
        ) .* π_θ
        kl_entropy_term = dropdims(sum(kl_entropy_term; dims=1); dims=1)

        loss = -(lr_term .- kl_entropy_term)
        gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st, state_t
end

function _fkl_gradient(
    up::DiscreteProximalMDCCEM,
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
    batch_size = size(states)[end]

    # Find the indices of the actions of maximal value
    ind = mapslices(x -> sortperm(x; rev=true), q; dims=1)
    top_ind = [CartesianIndex(ind[1, j], j) for j in 1:batch_size]

    lnπ_t, st_t = logprob(π, π_f, θ_t, state_t, states)
    π_t = exp.(lnπ_t)
    @tullio entropy_t[i] := -π_t[j, i] * lnπ_t[j, i]

    ς = 1 .+ up._τ .* (lnπ_t .+ reshape(entropy_t, 1, :))

    ∇π_θ = gradient(π_θ) do π_θ
        # Compute the gradient ∇J = 𝔼_{I*}[ln(π)]
        lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st, states)
        best_lnπ_θ = lnπ_θ[top_ind]

        loss = -(best_lnπ_θ .- sum(π_t .* (ς .- up._inv_λ) .* lnπ_θ; dims=1))
        gpu_mean(loss)
    end

    return ∇π_θ, π_st, qf_st, st_t
end

function _rkl_gradient(
    up::DiscreteProximalMDCCEM,
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
    batch_size = size(states)[end]

    # Find the indices of the actions of maximal value
    ind = mapslices(x -> sortperm(x; rev=true), q; dims=1)
    top_actions = [ind[1, j] for j in 1:batch_size]

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = zero(Float32, π_f)
    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t)
    lnπ_θ, π_st = logprob(π, π_f, π_θ, π_st)
    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]
            a_t = top_actions[i]

            lr_term = exp.(_∇ln_softmax_tabular(π_θ.layer_1, s_t, a_t) .- lnπ_t[a_t, s_t])

            ∇π_θ = _∇_softmax_tabular(π_θ.layer_1, s_t; sum_over_actions=false)
            kl_entropy_term = (
                (up._τ - up._inv_λ) .* lnπ_t[:, s_t] +
                up._inv_λ .* lnπ_θ[:, s_t]
            )
            kl_entropy_term = reshape(kl_entropy_term, 1, :)
            ∇π_θ .*= kl_entropy_term
            ∇π_θ_term = dropdims(sum(∇π_θ; dims=2); dims=2)

            g_i[:, s_t] .-= ((lr_term .- ∇π_θ_term) ./ batch_size)
            end
        g_i
    end


    return (gs,), π_st, qf_st, state_t
end

function _fkl_gradient(
    up::DiscreteProximalMDCCEM,
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
    batch_size = size(states)[end]

    # Find the indices of the actions of maximal value
    ind = mapslices(x -> sortperm(x; rev=true), q; dims=1)
    top_actions = [ind[1, j] for j in 1:batch_size]

    ####################################################################
    # Manual gradient calculation
    ####################################################################
    gs = spzeros(Float32, π_f)
    lnπ_t, state_t = logprob(π, π_f, θ_t, state_t)
    π_t = exp.(lnπ_t)

    entropy_t = sum(π_t .* lnπ_t; dims=1)

    ς = 1 .+ up._τ .* (lnπ_t .+ entropy_t)

    treemap!(gs) do g_i
        for i in 1:batch_size
            s_t = states[1, i]
            a_t = top_actions[i]

            # Compute expectation argument
            lr_term = _∇ln_softmax_tabular(π_θ.layer_1, s_t; sum_over_actions=false)

            ς_s_t = ς[:, s_t:s_t]'
            expectation_arg = (ς_s_t .- up._inv_λ) .* lr_term

            # Compute expectation 𝔼_{πₜ} [(ς(aₜ, sₜ) + 1/λ) ln(π(a | s, θ))]
            π_t_s_t = π_t[:, s_t:s_t]'
            expectation = sum(π_t_s_t .* expectation_arg; dims=2)
            expectation = dropdims(expectation; dims=2)

            g_i[:, s_t] .-= (
                _∇ln_softmax_tabular(π_θ.layer_1, s_t, a_t) .- expectation
            ) ./ batch_size
            end
        g_i
    end

    return (gs,), π_st, qf_st, state_t
end

