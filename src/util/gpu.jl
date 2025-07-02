"""
    gpu_mean(x; dims=1:ndims(x))

This function allows taking the mean of an array, while remaining differentiable on the GPU.

# Issue with differentiating `Statistics.mean` with `Zygote` on `CUDA`

The `mean` function from Statistics.jl doesn't play nicely with CUDA and Zygote. As simple
example, this code:

  julia> using CUDA
  julia> arr = cu(rand(100, 90))
  julia> gs = gradient(arr) do a
              mean(a)
          end

Will result in the following error

  ERROR: CUDA error: an illegal memory access was encountered (
      code 700, ERROR_ILLEGAL_ADDRESS)
  Stacktrace:
   [1] throw_api_error(res::CUDA.cudaError_enum)
     @ CUDA ~/.julia/packages/CUDA/YIj5X/lib/cudadrv/libcuda.jl:27
   [2] isdone
     @ ~/.julia/packages/CUDA/YIj5X/lib/cudadrv/stream.jl:111 [inlined]
   [3] spinning_synchronization(f::typeof(CUDA.isdone), obj::CuStream)
     @ CUDA ~/.julia/packages/CUDA/YIj5X/lib/cudadrv/synchronization.jl:79
   [4] device_synchronize(; blocking::Bool, spin::Bool)
     @ CUDA ~/.julia/packages/CUDA/YIj5X/lib/cudadrv/synchronization.jl:171
   [5] device_synchronize()
     @ CUDA ~/.julia/packages/CUDA/YIj5X/lib/cudadrv/synchronization.jl:169
   [6] top-level scope
     @ ~/.julia/packages/CUDA/YIj5X/src/initialization.jl:210

This `gpu_mean` function is a work-around for this issue.
"""
@inline function gpu_mean(x; dims=1:ndims(x))
    m = mean(x; dims=dims)
    return dims == 1:ndims(x) ? sum(m) : dropdims(m; dims=tuple(dims...))
end
