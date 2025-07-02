"""
    DiscreteCCEM

`DiscreteCCEM` implements the discrete-action Conditional Cross-Entropy Method (CCEM) for
policy improvement [1].

# References

[1] S. Neumann, Lim, S., Joseph, A., Pan, Y., White, A., White, M. Greedy
Actor-Critic: A New Conditional Cross-Entropy Method for Policy
Improvement. 2022.
"""
struct DiscreteCCEM <: AbstractActorUpdate
    # Actor Policy Entropy Regularization
    _τ::Float32         # Actor Temperature

    function DiscreteCCEM(τ::Real)
        @assert (τ >= 0) "expected τ >= 0"
        return new(τ)
    end
end

function setup(
    up::DiscreteCCEM,
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
)::UpdateState{DiscreteCCEM}
    return UpdateState(
        up,
        optim,
        (π_optim = Optimisers.setup(optim, π_θ),)
    )
end

function setup(
    up::DiscreteCCEM,
    ::AbstractEnvironment,
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
    error("cannot use DiscreteCCEM with simplex policies")
end

function update(
    st::UpdateState{DiscreteCCEM},
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
    q, qf_st = predict(qf, qf_f, qf_θ, qf_st, states)
    batch_size = size(states)[end]

    ∇π_θ, π_st, qf_st = _gradient(up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states)

    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    return UpdateState(
        st._update,
        st._optim,
        (π_optim = π_optim_state,),
    ), π_θ, π_st, qf_st
end

function _gradient(
    up::DiscreteCCEM,
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
    batch_size = size(states)[end]

    # Find the indices of the actions of maximal value
    ind = mapslices(x -> sortperm(x; rev=true), q; dims=1)
    ind = ind[1, :]
    top_ind = CartesianIndex.(ind, LinearIndices(ind))

    ∇π_θ = gradient(π_θ) do π_θ
        # Compute the gradient ∇J = 𝔼_{I*}[ln(π)]
        lnπ, π_st = logprob(π, π_f, π_θ, π_st, states)
        if up._τ > 0
            # Compute entropy regularization
            H, π_st = entropy(π, π_f, π_θ, π_st, states)
            -gpu_mean(lnπ[top_ind]) - up._τ * gpu_mean(H)
        else
            -gpu_mean(lnπ[top_ind])
        end
    end

    return ∇π_θ, π_st, qf_st
end
