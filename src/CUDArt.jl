module CUDArt

export
    # pointer symbols
    CudaPtr, convert, rawpointer, CUDA_NULL, integer, unsigned,
    eltype, ==, -, +, zero, one,
    # array symbols
    AbstractCudaArray, AbstractCudaVector, AbstractCudaMatrix,
    CudaArray, CudaVector, CudaMatrix, CudaVecOrMat,
    CudaPitchedArray, HostArray,
    to_host, similar, copy,
    # other symbols
    device, devices, device_reset, attribute, capability,
    driver_version, runtime_version,
    CuModule, CuFunction, unload,
    pitchel, pitchbytes,
    launch, device_synchronize, synchronize,
    Stream, null_stream, cudasleep,
    destroy, free, cudafinalizer

import Base: length, size, ndims, eltype, similar, pointer, stride,
    convert, reinterpret, show, copy,
    copy!, get!, fill!, wait

# Prepare the CUDA runtime API bindings
include("libcudart.jl")
import .CUDArt_gen
const rt = CUDArt_gen

# To load PTX code, we also need access to the driver API module utilities
const libcuda = find_library(["libcuda"], ["/usr/local/cuda"])
if isempty(libcuda)
    error("CUDA driver API library cannot be found")
end

include("version.jl")
include("types.jl")
include("device.jl")
include("stream.jl")
#include("event.jl")
include("module.jl")
include("pointer.jl")
include("arrays.jl")
include("execute.jl")

end
