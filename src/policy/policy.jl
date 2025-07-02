export
    AbstractPolicy,
    AbstractParameterisedPolicy,
    AbstractDiscreteParameterisedPolicy,
    AbstractContinuousParameterisedPolicy,
    AbstractStaticPolicy,
    sample,
    rsample,
    continuous,
    discrete,
    logprob,
    IIDPolicy,

    # Bounded Policies
    BoundedPolicy,
    BetaPolicy,
    LogitNormalPolicy,
    ArctanhNormalPolicy,
    KumaraswamyPolicy,

    # Truncated Policies
    TruncatedPolicy,
    TruncatedNormalPolicy,
    TruncatedLaplacePolicy,

    # UnBounded Policies
    UnBoundedPolicy,
    LogisticPolicy,
    GammaPolicy,
    LaplacePolicy,
    NormalPolicy,

    # Static Policies
    StaticUniformPolicy,

    # DiscretePolicies
    SoftmaxPolicy,
    SimplexPolicy

const _fEPSILON = 1f-3 # This is a bit aggressive

import MLUtils: batch

include("abstract_policy.jl")
include("iid_policy.jl")

# TODO: eventually, we can actually use the implemenation in the gpu_iid_X.jl files for both
# CPU and GPU, because it will work on both devices. It will require a
# @_delegate_extended_distributions macro call for each policy distribution that we want to
# support
#
# For many distributions, we cannot differentiate the logpdf function in
# Distributions.jl using Zygote and CUDA. For some reason, if we construct a distribution
# within a gradient(ps) do ps ... end statement, CUDA complains that it only wants variables
# for which memory can be aligned. I believe this is actually due to a different underlying
# error (since all variables in these cases can be memory aligned) occurring, and then whent
# his error is raised, CUDA just raises the "memory inline" error, hiding the underlying
# problem...
#
# This can happen if e.g. string manipulation occurs.
function logprob_function end
function prob_function end
function kl_function end
function quantile_function end
function cdf_function end
function ccdf_function end

# Only distributions which have this function defined can be used with GPUs
macro _delegate_extended_distributions(D, pre)
    logpdf = Symbol(pre, "logpdf")
    pdf = Symbol(pre, "pdf")
    kldiv = Symbol(pre, "kldivergence")
    quantile = Symbol(pre, "quantile")
    cdf = Symbol(pre, "cdf")
    ccdf = Symbol(pre, "ccdf")
    return quote
        $ActorCritic.logprob_function(::IIDPolicy{V,$D}) where {V} = $logpdf
        $ActorCritic.prob_function(::IIDPolicy{V,$D}) where {V} = $pdf
        $ActorCritic.kl_function(::IIDPolicy{V,$D}) where {V} = $kldiv
        $ActorCritic.quantile_function(::IIDPolicy{V,$D}) where {V} = $quantile
        $ActorCritic.cdf_function(::IIDPolicy{V,$D}) where {V} = $cdf
        $ActorCritic.ccdf_function(::IIDPolicy{V,$D}) where {V} = $ccdf
    end
end

@_delegate_extended_distributions Beta beta
@_delegate_extended_distributions Normal norm
@_delegate_extended_distributions ArctanhNormal atanhnorm
@_delegate_extended_distributions Laplace laplace
@_delegate_extended_distributions Kumaraswamy kumaraswamy
@_delegate_extended_distributions LogitNormal logitnorm
@_delegate_extended_distributions Logistic logistic

include("bounded_policy.jl")
include("truncated_policy.jl")
include("unbounded_policy.jl")
include("iid_sampling.jl")
include("normal.jl")
include("laplace.jl")
include("arctanhnormal.jl")
include("logitnormal.jl")
include("gpu_iid_policy.jl")
include("gpu_iid_sampling.jl")
include("static_policy.jl")
include("softmax_policy.jl")
include("simplex_policy.jl")
