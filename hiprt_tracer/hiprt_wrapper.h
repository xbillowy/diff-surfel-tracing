/**
 * @file hiprt_wrapper.h
 * @brief HIPRT state wrapper: the ROCm/HIP analogue of optix_tracer/optix_wrapper.h.
 *
 * This header is the boundary between the torch translation unit (trace_surfels.cpp,
 * compiled by torch's CUDAExtension with the real HIP runtime in scope) and the
 * HIPRT/Orochi translation unit (hiprt_wrapper.cpp, compiled standalone with
 * Orochi's hipew loader). The two cannot share the same TU: Orochi's hipew driver
 * loader redeclares the HIP driver API and conflicts with torch's <hip/hip_runtime.h>.
 * So this interface exposes ONLY plain C++ / POD / void* entry points; no Orochi or
 * HIPRT types leak across it. The public class name OptiXStateWrapper is preserved so
 * ext.cpp and diff_surfel_tracing/__init__.py are unchanged.
 */

#pragma once

#include <string>


// Opaque HIPRT/Orochi state, fully defined only inside hiprt_wrapper.cpp.
struct OptiXState;


// Same public name as the OptiX build for binding/source compatibility.
class OptiXStateWrapper
{
public:
    OptiXState* optixState;

    OptiXStateWrapper(const std::string& pkg_dir);
    ~OptiXStateWrapper();
};


// ---------------------------------------------------------------------------
// HIPRT tracer C++ interface (implemented in hiprt_wrapper.cpp). All arguments
// are POD / raw device pointers / a stream-as-void*; no HIPRT or Orochi type is
// visible to the torch TU that calls these.
// ---------------------------------------------------------------------------

// Build (or refit) the surfel-disk triangle BVH. vertices/triangles are device
// pointers to (num_vertices, 3) float and (num_triangles, 3) int32 buffers.
void hiprtTracerBuildGeometry(
    OptiXStateWrapper& wrapper,
    const float* d_vertices, int num_vertices,
    const int* d_triangles, int num_triangles,
    unsigned int rebuild, void* hip_stream);

// Launch the forward (which_kernel=0) or backward (which_kernel=1) trace kernel.
// h_params points at a host Params struct of params_size bytes; the wrapper
// device-copies it and launches over the (H, W) grid on hip_stream.
void hiprtTracerLaunch(
    const OptiXStateWrapper& wrapper, int which_kernel,
    const void* h_params, unsigned long params_size,
    int H, int W, void* hip_stream);
