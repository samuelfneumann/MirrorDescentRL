abstract type AbstractAgent end

function select_action(agent::AbstractAgent, s_t)
    error("select_action not implemented for $(typeof(agent))")
end

function start!(agent::AbstractAgent, s_0)::Nothing
    error("start! not implemented for $(typeof(agent))")
end

function step!(agent::AbstractAgent, s_t, a_t, r_tp1, s_tp1, γ_tp1)::Nothing
    error("step! not implemented for $(typeof(agent))")
end

function stop!(agent::AbstractAgent, r_T, s_T, γ_T)::Nothing
    error("stop! not implemented for $(typeof(agent))")
end

"""
    train!(::AbstractAgent)
    train!(::AbstractPolicy)

Set the argument to training mode
"""
function train! end

"""
    eval!(::AbstractAgent)
    eval!(::AbstractPolicy)

Set the argument to evaluation mode
"""
function eval! end

