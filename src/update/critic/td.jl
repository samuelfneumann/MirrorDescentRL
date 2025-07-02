struct TD <: AbstractStateValueCriticUpdate end

function setup(
    up::TD,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    vf::AbstractStateValueFunction,
    vf_f,
    vf_θ,
    vf_st,
    optim::Optimisers.AbstractRule,
    ::AbstractRNG,
)::UpdateState
    return UpdateState(up, optim, (optim = Optimisers.setup(optim, vf_θ),))
end

function update(
    st::UpdateState{TD},
    _::AbstractParameterisedPolicy,
    _π_f,
    _π_θ,
    _π_st,
    vf::AbstractStateValueFunction,
    vf_f,
    vf_θ,
    vf_st,
    vf_target_θ,
    vf_target_st,
    s_t,
    r_tp1,
    s_tp1,
    γ_tp1,
)
    up = st._update

    # Compute the (regularised) target using the target network
    v_tp1, vf_target_st = predict(vf, vf_f, vf_target_θ, vf_target_st, s_tp1)
    target = r_tp1 .+ γ_tp1 .* v_tp1

    return (update(st, vf, vf_f, vf_θ, vf_st, s_t, target)..., vf_target_st, _π_st)
end

function update(
    st::UpdateState{TD},
    vf::AbstractStateValueFunction,
    vf_f,
    vf_θ,
    vf_st,
    s_t,
    target,
)
    up = st._update

    ∇v_θ = gradient(vf_θ) do θ
        v_t, vf_st = predict(vf, vf_f, θ, vf_st, s_t)
        gpu_mean((v_t .- target) .^ 2)
    end

    v_optim_state = st._state.optim
    v_optim_state, vf_θ = Optimisers.update(v_optim_state, vf_θ, only(∇v_θ))

    return UpdateState(
        st._update,
        st._optim,
        (optim = v_optim_state,),
    ), vf_θ, vf_st
end
