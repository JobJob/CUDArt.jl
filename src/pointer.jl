# pointer.jl
#
# Julia type and memory management for CUDA pointers.  Adapted and extracted
# from JuliaGPU/CUDArt.jl/src/arrays.jl and julia/base/pointer.jl.
#
# Author: Nick Henderson <nwh.stanford.edu>
# Created: 2014-09-04
# License: MIT
#

# A raw CUDA pointer
type CudaPtr{T}
    ptr::Ptr{T}
end

# Type alias for previous name
typealias CudaDevicePtr CudaPtr

#############################
# Low-level memory handling #
#############################

CudaPtr() = CudaPtr(C_NULL)
CudaPtr(T::Type) = CudaPtr(convert(Ptr{T},C_NULL))
convert{T}(::Type{Ptr{T}}, p::CudaPtr{T}) = p.ptr
convert{T}(::Type{Ptr{None}}, p::CudaPtr{T}) = convert(Ptr{None}, p.ptr)

rawpointer(p::CudaPtr) = p

# Enable both manual and garbage-collected memory management.
# If you need to free resources, you can call free manually.
# cuda_ptrs keeps track of all memory that needs to be freed,
# and prevents double-free (which otherwise causes serious problems).
# key = ptr, val = device id
const cuda_ptrs = Dict{Any,Int}()

function malloc(T::Type, n::Integer)
    p = Ptr{Void}[C_NULL]
    nbytes = sizeof(T)*n
    rt.cudaMalloc(p, nbytes)
    cptr = CudaPtr(convert(Ptr{T},p[1]))
    finalizer(cptr, free)
    cuda_ptrs[cptr] = device()
    cptr
end
malloc(nbytes::Integer) = malloc(Uint8, nbytes)

function free{T}(p::CudaPtr{T})
    cnull = convert(Ptr{T}, C_NULL)
    if p.ptr != cnull && haskey(cuda_ptrs, p)
        delete!(cuda_ptrs, p)
        rt.cudaFree(p)
        p.ptr = cnull
    end
end

typealias Ptrs Union(Ptr, CudaPtr, rt.cudaPitchedPtr)
typealias CudaPtrs Union(CudaPtr, rt.cudaPitchedPtr)

cudamemcpykind(dstp::Ptr, srcp::Ptr) = rt.cudaMemcpyHostToHost
cudamemcpykind(dstp::CudaPtrs, srcp::Ptr) = rt.cudaMemcpyHostToDevice
cudamemcpykind(dstp::Ptr, srcp::CudaPtrs) = rt.cudaMemcpyDeviceToHost
cudamemcpykind(dstp::CudaPtrs, srcp::CudaPtrs) = rt.cudaMemcpyDeviceToDevice
cudamemcpykind(dst::Ptrs, src::Ptrs) = error("This should never happen") # prevent a useless ambiguity warning
cudamemcpykind(dst, src::Ptrs) = cudamemcpykind(pointer(dst), src)
cudamemcpykind(dst::Ptrs, src) = cudamemcpykind(dst, pointer(src))
cudamemcpykind(dst, src) = cudamemcpykind(pointer(dst), pointer(src))

## converting pointers to an appropriate unsigned ##

const CUDA_NULL = CudaPtr()

# pointer to integer
convert(::Type{Uint}, x::CudaPtr) = convert(Uint,x.ptr)
convert{T<:Integer}(::Type{T}, x::CudaPtr) = convert(T,unsigned(x))

# integer to pointer
convert{T}(::Type{CudaPtr{T}}, x::Integer) = CudaPtr(convert(Ptr{T},x))

# pointer to pointer
convert{T}(::Type{CudaPtr{T}}, p::CudaPtr{T}) = p
convert{T}(::Type{CudaPtr{T}}, p::CudaPtr) = CudaPtr(convert(Ptr{T},p.ptr))

# object to pointer
#convert(::Type{Ptr{Uint8}}, x::Symbol) = ccall(:jl_symbol_name, Ptr{Uint8}, (Any,), x)
#convert(::Type{Ptr{Int8}}, x::Symbol) = ccall(:jl_symbol_name, Ptr{Int8}, (Any,), x)
#convert(::Type{Ptr{Uint8}}, s::ByteString) = convert(Ptr{Uint8}, s.data)
#convert(::Type{Ptr{Int8}}, s::ByteString) = convert(Ptr{Int8}, s.data)

#convert{T}(::Type{Ptr{T}}, a::Array{T}) = ccall(:jl_array_ptr, Ptr{T}, (Any,), a)
#convert(::Type{Ptr{None}}, a::Array) = ccall(:jl_array_ptr, Ptr{None}, (Any,), a)

# note: these definitions don't mean any AbstractArray is convertible to
# pointer. they just map the array element type to the pointer type for
# convenience in cases that work.
#pointer{T}(x::AbstractArray{T}) = convert(Ptr{T},x)
#pointer{T}(x::AbstractArray{T}, i::Integer) = convert(Ptr{T},x) + (i-1)*elsize(x)

# unsafe pointer to array conversions
#pointer_to_array(p, d::Integer, own=false) = pointer_to_array(p, (d,), own)
#function pointer_to_array{T,N}(p::Ptr{T}, dims::NTuple{N,Int}, own::Bool=false)
#    ccall(:jl_ptr_to_array, Array{T,N}, (Any, Ptr{Void}, Any, Int32),
#          Array{T,N}, p, dims, own)
#end
#function pointer_to_array{T,N}(p::Ptr{T}, dims::NTuple{N,Integer}, own::Bool=false)
#    for d in dims
#        if !(0 <= d <= typemax(Int))
#            error("invalid Array dimensions")
#        end
#    end
#    pointer_to_array(p, convert((Int...), dims), own)
#end
#unsafe_load(p::Ptr,i::Integer) = pointerref(p, int(i))
#unsafe_load(p::Ptr) = unsafe_load(p, 1)
#unsafe_store!(p::Ptr{Any}, x::ANY, i::Integer) = pointerset(p, x, int(i))
#unsafe_store!{T}(p::Ptr{T}, x, i::Integer) = pointerset(p, convert(T,x), int(i))
#unsafe_store!{T}(p::Ptr{T}, x) = pointerset(p, convert(T,x), 1)

# convert a raw Ptr to an object reference, and vice-versa
#unsafe_pointer_to_objref(p::Ptr) = pointertoref(unbox(Ptr{Void},p))
#pointer_from_objref(x::Any) = ccall(:jl_value_ptr, Ptr{Void}, (Any,), x)

integer(x::CudaPtr) = convert(Uint, x.ptr)
unsigned(x::CudaPtr) = convert(Uint, x.ptr)

eltype{T}(::CudaPtr{T}) = T

## limited pointer arithmetic & comparison ##

==(x::CudaPtr, y::CudaPtr) = uint(x) == uint(y)
-(x::CudaPtr, y::CudaPtr) = uint(x) - uint(y)

+(x::CudaPtr, y::Integer) = oftype(x, uint(uint(x) + y))
-(x::CudaPtr, y::Integer) = oftype(x, uint(uint(x) - y))
+(x::Integer, y::CudaPtr) = y + x

zero{T}(::Type{CudaPtr{T}}) = convert(CudaPtr{T}, 0)
zero{T}(x::CudaPtr{T}) = convert(CudaPtr{T}, 0)
one{T}(::Type{CudaPtr{T}}) = convert(CudaPtr{T}, 1)
one{T}(x::CudaPtr{T}) = convert(CudaPtr{T}, 1)
