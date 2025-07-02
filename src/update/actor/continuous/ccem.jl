"""
    ContinuousCCEM

ContinuousCCEM implements the Conditional Cross-Entropy Method (CCEM) for policy improvement
[1] with number of samples `n` and action percentiles `ρ` and `ρ̃` for the actor and proposal
policy updates respectively.

## Updates

This section discusses how the actor and proposal policies are updated.

The actor and proposal policies, denoted as `π` and `π̃` respectively, use a CCEM update for
policy improvement. The CCEM update works as follows:
    1. Set hyperparameters `n`, `ρ`, `ρ̃`
    2. Generate a set of `n` actions 𝔸
    3. Order that set of actions using some metric, `Q`. Typically `Q` is an action-value
       function. Denote this set `I = { a₁, a₂, a₃, ... aₙ}` with indices such that for
           `i > j, Q(aᵢ) > Q(aⱼ)`
    4. Update the actor policy `π` by increasing the log-likelihood of the `⌊ρn⌋` actions of
       maximum value under `Q`. The effective gradient is

            I* = { a | Q(a) > Q(a_{⌊ρn⌋}) }, where a_{⌊ρn⌋} is the ρ-th action percentile
                under Q
            ∇ = 𝔼_{a ∼ I*} [∇ln(π(a|s)]

    5. Update the proposal policy `π̃` by increasing the log-likelihood of the `⌊ρ̃n⌋` actions
       of maximum value under `Q`.

            Ĩ* = { a | Q(a) > Q(a_{⌊ρ̃n⌋}) }, where a_{⌊ρ̃n⌋} is the ρ̃-th action percentile
                under Q
            ∇ = 𝔼_{a ∼ Ĩ*} [∇ln(π̃(a|s)]


The set of actions for the CCEM update in (2) above are drawn from the proposal policy as in
[1].

The proposal policy, denoted as π̃, is updated using a CCEM update as well. To generate
action samples for the CCEM update, π̃ itself is sampled. This sample is shared for updating
both the proposal policy and actor policy.

# References
[1] S. Neumann, Lim, S., Joseph, A., Pan, Y., White, A., White, M. Greedy
Actor-Critic: A New Conditional Cross-Entropy Method for Policy
Improvement. 2022.

[2] Greedification Operators for Policy Optimization: Investigating Forward
and Reverse KL Divergences. Chan, A., Silva H., Lim, S., Kozuno, T.,
Mahmood, A. R., White, M. 2021.

[3] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor. Haarnoja, T., Zhou, A., Abbeel, P.,
Levine, S. International Conference on Machine Learning. 2018.

[4] Soft Actor-Critic: Algorithms and Applications. Haarnoja, T.,
Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V.,
Zhu, H., Gupta, A., Abbeel, P., Levine, S. In preparation. 2019.
"""
struct ContinuousCCEM <: AbstractActorUpdate
    _n::Int
    _ρ::Float32
    _ρ̃::Float32

    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature
    _π_num_entropy_samples::Int

    # Proposal Policy Entropy Regularization
    _τ̃::Float32         # Proposal Temperature
    _π̃_num_entropy_samples::Int

    function ContinuousCCEM(
        n::Int, ρ::Real, ρ̃::Real, τ::Real, π_num_entropy_samples::Int, τ̃::Real,
        π̃_num_entropy_samples::Int,
    )
        @assert (τ >= 0) "expected τ >= 0"
        @assert (τ̃ >= 0) "expected τ̃ >= 0"
        @assert (π_num_entropy_samples >= 0) "expected π_num_entropy_samples >= 0"
        @assert (π̃_num_entropy_samples >= 0) "expected π̃_num_entropy_samples >= 0"
        @assert (ρ >= 0) "expected ρ >= 0"
        @assert (ρ̃ >= 0) "expected ρ̃ >= 0"
        @assert (trunc(n * ρ) > 0) "expected ⌊ρn⌋ > 0"
        @assert (trunc(n * ρ̃) > 0) "expected ⌊ρ̃n⌋ > 0"

        return new(n, ρ, ρ̃, τ, π_num_entropy_samples, τ̃, π̃_num_entropy_samples)
    end
end

function setup(
    up::ContinuousCCEM,
    ::AbstractEnvironment,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    qf::AbstractActionValueFunction,
    qf_f,   # q function model
    qf_θ,   # q function model parameters
    qf_st,  # q function model state
    optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{ContinuousCCEM}
    π̃ = deepcopy(π)
    π̃_f = deepcopy(π_f)
    π̃_θ = deepcopy(π_θ)
    π̃_st = deepcopy(π_st)
    π̃_optim = deepcopy(optim)

    return setup(
        up, π, π_f, π_θ, π_st, π̃, π̃_f, π̃_θ, π̃_st, qf, qf_f, qf_θ, qf_st, optim, π̃_optim,
        rng,
    )
end

function setup(
    up::ContinuousCCEM,
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    π̃::AbstractContinuousParameterisedPolicy,
    π̃_f,    # proposal policy model
    π̃_θ,    # proposal policy model parameters
    π̃_st,   # proposal policy model state
    qf::AbstractActionValueFunction,
    qf_f,
    qf_θ,
    qf_st,
    π_optim::Optimisers.AbstractRule,
    π̃_optim::Optimisers.AbstractRule,
    rng::AbstractRNG,
)::UpdateState{ContinuousCCEM}
    return UpdateState(
        up,
        π_optim,
        (
            π_optim = Optimisers.setup(π_optim, π_θ),
            π̃_optim = Optimisers.setup(π̃_optim, π̃_θ),
            π̃ = π̃,
            π̃_f = π̃_f,
            π̃_θ = π̃_θ,
            π̃_st = π̃_st,
            rng = Lux.replicate(rng)
        )
    )
end

function update(
    st::UpdateState{ContinuousCCEM},
    π::AbstractContinuousParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    qf::Q,
    qf_f,
    qf_θ,
    qf_st,
    states::AbstractArray, # Must be >= 2D
)
    up = st._update
    rng = Lux.replicate(st._state.rng)

    # Unpack proposal policy from update state
    π̃ = st._state.π̃
    π̃_f = st._state.π̃_f
    π̃_θ = st._state.π̃_θ
    π̃_st = st._state.π̃_st

    # Get the actions of maximal value as ordered by the action-value function critic
    π_top_actions, π̃_top_actions, π̃_st = _get_top_actions(
        rng, states, π̃, π̃_f, π̃_θ, π̃_st, qf, qf_f, qf_θ, qf_st, up._n, up._ρ, up._ρ̃
    )

    # Compute the actor and proposal gradients using the actions of maximal value gotten
    # above
    ∇π_θ, π_st = _gradient(
        up, rng, π, π_f, π_θ, π_st, up._τ, up._π_num_entropy_samples, states, π_top_actions,
    )
    ∇π̃_θ, π̃_st = _gradient(
        up, rng, π̃, π̃_f, π̃_θ, π̃_st, up._τ̃, up._π̃_num_entropy_samples, states, π̃_top_actions
    )

    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))
    π̃_optim_state = st._state.π̃_optim
    π̃_optim_state, π̃_θ = Optimisers.update(π̃_optim_state, π̃_θ, only(∇π̃_θ))

    return UpdateState(
        st._update,
        st._optim,
        (
            π_optim = π_optim_state,
            π̃_optim = π̃_optim_state,
            π̃ = π̃,
            π̃_f = π̃_f,
            π̃_θ = π̃_θ,
            π̃_st = π̃_st,
            rng = rng
        )
    ), π_θ, π_st, qf_st
end

function _gradient(
    ::ContinuousCCEM,
    rng::AbstractRNG,
    π::AbstractContinuousParameterisedPolicy,
    π_f,
    π_θ,
    π_st,
    τ,
    π_num_entropy_samples,
    states::AbstractArray,
    top_actions,
)
    ∇π_θ = if τ > 0 && π_num_entropy_samples > 0
        entropy_samples, π_st = sample(
            π, rng, π_f, π_θ, π_st, states; num_samples=π_num_entropy_samples,
        )

        gradient(π_θ) do θ
            params, π_st = get_params(π, π_f, θ, π_st, states)
            lnπ = logprob(π, top_actions, params)

            # Compute entropy regularization
            entropy_lnπ = logprob(π, entropy_samples, params)
            entropy = -(entropy_lnπ .^ 2) ./ 2

            -gpu_mean(lnπ) - τ * gpu_mean(entropy)
        end
    else
        gradient(π_θ) do θ
            lnπ, π_st = logprob(π, π_f, θ, π_st, states, top_actions)
            -gpu_mean(lnπ)
        end
    end

    return ∇π_θ, π_st
end

function _get_top_actions(batched_actions, ind)
    top_ind = CartesianIndex.(ind, reshape(LinearIndices(ind[1, :]), 1, :))
    top_actions = @inbounds batched_actions[:, top_ind]
    return top_actions
end

function _get_top_actions(
    rng::AbstractRNG,
    states::AbstractArray,
    π̃,
    π̃_f,
    π̃_θ,
    π̃_st,
    qf,
    qf_f,
    qf_θ,
    qf_st,
    n,
    ρ,
    ρ̃,
)
    batch_size = size(states)[end]
    state_size = size(states)[begin:end-1]

    # Sample actions from π̃ for the ContinuousCCEM update
    batched_actions, π̃_st = sample(
        π̃,
        rng,
        π̃_f,
        π̃_θ,
        π̃_st,
        states;
        num_samples=n,
    )  # (𝒜, num_samples, batch)

    action_size = size(batched_actions, 1)
    actions = reshape(batched_actions, action_size..., :)

    # Stack states to calculate action values. Each sampled action requires one state
    # observation
    stacked_states = repeat(
        states;
        inner=(ones(Int, length(state_size))..., n),
    )

    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, stacked_states, actions)
    q = reshape(q, n, batch_size)

    # Calculate how many actions of maximal value to increase the log-likelihood of
    π_n = trunc(Int, ρ * n)
    π̃_n = trunc(Int, ρ̃ * n)

    # Find the indices of actions of maximal value, only sorting the number of actions
    # absolutely required
    n = max(π_n, π̃_n)
    ixs = if CUDA.functional()
        [CuArray{Int32}(undef, size(q, 1)) for _ in 1:size(q, 2)]
    else
        [Array{Int}(undef, size(q, 1)) for _ in 1:size(q, 2)]
    end
    sortperm!.(ixs, eachcol(q))
    ind = reduce(hcat, ixs)[end-n+1:end, :]

    _top_actions = _get_top_actions(batched_actions, ind)
    π_top_actions, π̃_top_actions = if π_n > π̃_n
        _top_actions, _top_actions[:, end-π̃_n+1:end, :]
    else
        _top_actions[:, end-π_n+1:end, :], _top_actions
    end

    return π_top_actions, π̃_top_actions, π̃_st
end
