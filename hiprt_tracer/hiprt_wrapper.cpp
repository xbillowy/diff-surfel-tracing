/**
 * @file hiprt_wrapper.cpp
 * @brief HIPRT context + JIT-kernel setup and launch. ROCm/HIP analogue of
 *        optix_wrapper.cpp. Compiled STANDALONE (Orochi's hipew loader, no torch
 *        headers) so it does not clash with torch's <hip/hip_runtime.h>; the torch
 *        side reaches it through the POD/void* interface in hiprt_wrapper.h.
 *
 * Where OptiX loaded offline PTX and built a pipeline/SBT, this JIT-compiles the
 * trace kernels (kernels.h) at runtime via HIPRT (Orochi -> hiprtc), registers the
 * hit-collection filter functor in a func table, builds the surfel-disk triangle
 * BVH, and launches the kernels.
 */

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>

#include "hiprt_wrapper.h"


#define HIPRT_CHECK(c)                                                                   \
    {                                                                                    \
        hiprtError _e = (c);                                                             \
        if (_e != hiprtSuccess)                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << " HIPRT error " << _e << "\n";   \
    }
#define ORO_CHECK(c)                                                                     \
    {                                                                                    \
        oroError _e = (c);                                                              \
        if (_e != oroSuccess) {                                                          \
            const char* _m = nullptr;                                                    \
            oroGetErrorString(_e, &_m);                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << " Orochi error " << _e           \
                      << " (" << (_m ? _m : "?") << ")\n";                               \
        }                                                                                \
    }


// HIPRT/Orochi state for one tracer instance.
struct OptiXState
{
    oroDevice device;
    oroCtx ctx = nullptr;
    hiprtContext context = nullptr;

    hiprtGeometry geom = nullptr;
    void* d_geom_temp = nullptr;
    size_t geom_temp_size = 0;
    bool has_geom = false;

    oroFunction forward_func = nullptr;
    oroFunction backward_func = nullptr;
    hiprtFuncTable func_table = nullptr;
};


static std::string readFileContent(const std::string& path)
{
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ERROR: HIPRT tracer could not open kernel source " << path << "\n";
        return std::string();
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}


// HIP grid covers the (H, W) launch the OptiX tracer used (launch dims (H,W,1)).
static const int TRACE_BLOCK_X = 8;
static const int TRACE_BLOCK_Y = 8;


OptiXStateWrapper::OptiXStateWrapper(const std::string& pkg_dir)
{
    optixState = new OptiXState();
    OptiXState* st = optixState;

    // On Windows, the AMD GPU display driver installs amdhip64_7.dll into
    // System32, which LoadLibraryA finds before the ROCm SDK's copy (System32
    // is ahead of PATH in the legacy DLL search order). Orochi's GLUE must use
    // the SAME amdhip64_7.dll as torch (the ROCm SDK version in _rocm_sdk_core)
    // so that oroCtxGetCurrent returns torch's context and oroGetDeviceProperties
    // returns the correct device arch for hiprtc JIT compilation. Build the full
    // path from ROCM_PATH or HIP_PATH; fall back to the bare name if unset.
    // The hiprtc DLL's version digits change layout between ROCm releases
    // (hiprtc0605.dll, hiprtc07013.dll), so take the copy the SDK ships.
    const char* hip_custom[2] = { nullptr, nullptr };
    const char* hiprtc_custom[2] = { nullptr, nullptr };
    std::string hip_full_path_glue, hiprtc_full_path_glue;
#ifdef _WIN32
    const char* _rocm_init = getenv("ROCM_PATH");
    if (!_rocm_init) _rocm_init = getenv("HIP_PATH");
    if (_rocm_init) {
        hip_full_path_glue = std::string(_rocm_init) + "\\bin\\amdhip64_7.dll";
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(std::filesystem::path(_rocm_init) / "bin", ec)) {
            if (e.path().filename().string().rfind("hiprtc0", 0) == 0 && e.path().extension() == ".dll")
                hiprtc_full_path_glue = e.path().string();
        }
        hip_custom[0] = hip_full_path_glue.c_str();
        if (!hiprtc_full_path_glue.empty())
            hiprtc_custom[0] = hiprtc_full_path_glue.c_str();
    }
#endif
    const char** hip_paths_arg = (hip_custom[0] != nullptr) ? hip_custom : nullptr;
    const char** hiprtc_paths_arg = (hiprtc_custom[0] != nullptr) ? hiprtc_custom : nullptr;
    if (oroInitialize(ORO_API_HIP, 0, hip_paths_arg, hiprtc_paths_arg) != 0) {
        std::cerr << "ERROR: oroInitialize(HIP) failed; HIPRT tracer unavailable\n";
        return;
    }
    ORO_CHECK(oroInit(0));

    // Bind Orochi to the SAME HIP context torch is using, so the BVH builder and
    // the trace kernels see torch's device pointers. torch has already created and
    // made-current the device's HIP primary context by the time this runs, so
    // oroCtxGetCurrent returns it. Creating a fresh context with oroCtxCreate
    // instead would put the BVH build in a different context and the surfel-vertex
    // pointers would be invalid there (silent: a degenerate BVH, zero hits).
    int hip_dev = 0;
    oroGetDevice(&hip_dev);
    ORO_CHECK(oroDeviceGet(&st->device, hip_dev));
    oroError ge = oroCtxGetCurrent(&st->ctx);
    if (ge != oroSuccess || st->ctx == nullptr) {
        // Fall back to retaining the device primary context (still torch's).
        ORO_CHECK(oroDevicePrimaryCtxRetain(reinterpret_cast<oroCtx_t*>(&st->ctx), st->device));
        ORO_CHECK(oroCtxSetCurrent(st->ctx));
    }

    hiprtContextCreationInput ci{};
    ci.deviceType = hiprtDeviceAMD;
    ci.ctxt = oroGetRawCtx(st->ctx);
    ci.device = oroGetRawDevice(st->device);
    HIPRT_CHECK(hiprtCreateContext(HIPRT_API_VERSION, ci, st->context));
    HIPRT_CHECK(hiprtSetLogLevel(st->context, hiprtLogLevelError | hiprtLogLevelWarn));

    std::string cache_dir = pkg_dir + "/hiprt_cache";
#ifdef _WIN32
    std::string mkdir_cmd = "mkdir \"" + cache_dir + "\" 2>nul";
#else
    std::string mkdir_cmd = "mkdir -p '" + cache_dir + "'";
#endif
    (void)system(mkdir_cmd.c_str());
    HIPRT_CHECK(hiprtSetCacheDirPath(st->context, cache_dir.c_str()));

    std::string kernel_path = pkg_dir + "/kernels.h";
    std::string src = readFileContent(kernel_path);
    if (src.empty()) {
        std::cerr << "ERROR: HIPRT tracer kernel source empty/missing at " << kernel_path << "\n";
        return;
    }

    // JIT include paths: the package dir (kernels.h + its includes) and the HIP
    // headers (kernels.h pulls hip/hip_runtime.h for float3/atomicAdd/etc.; the
    // runtime compiler does not add the ROCm include dir by default). ROCM_PATH /
    // HIP_PATH override the default /opt/rocm install location.
    std::string incPkg = std::string("-I") + pkg_dir;
    const char* rocm_env = getenv("ROCM_PATH");
    if (!rocm_env) rocm_env = getenv("HIP_PATH");
    std::string rocm_path = rocm_env ? rocm_env : "/opt/rocm";
    std::string incHip = std::string("-I") + rocm_path + "/include";
    // USE_ROCM selects the HIP side of the headers shared with the OptiX build
    // (params.h, auxiliary.h). The runtime compiler does not define it itself.
    std::vector<const char*> opts = { incPkg.c_str(), incHip.c_str(), "-DUSE_ROCM=1" };
    // Cap the per-thread VGPR budget by declaring the launch block size (we launch
    // 8x8=64). Without this, the hiprtc/comgr JIT (ROCm 7.2.x) miscompiles these
    // register-heavy traversal kernels -- values that cross the chunk-traversal
    // region (the chunk count, the per-hit transform outputs) read stale/zero/NaN.
    // Declaring the small block size relaxes the VGPR limit and makes codegen
    // correct.
    opts.push_back("--gpu-max-threads-per-block=64");

    hiprtFuncNameSet fns{};
    fns.filterFuncName = "surfelFilter";
    std::vector<hiprtFuncNameSet> fnsv = { fns };

    const char* fwd_name = "forward_kernel";
    HIPRT_CHECK(hiprtBuildTraceKernels(
        st->context, 1, &fwd_name, src.c_str(), "kernels.h",
        0, nullptr, nullptr, (uint32_t)opts.size(), opts.data(),
        1, 1, fnsv.data(),
        reinterpret_cast<hiprtApiFunction*>(&st->forward_func), nullptr, true));

    const char* bwd_name = "backward_kernel";
    HIPRT_CHECK(hiprtBuildTraceKernels(
        st->context, 1, &bwd_name, src.c_str(), "kernels.h",
        0, nullptr, nullptr, (uint32_t)opts.size(), opts.data(),
        1, 1, fnsv.data(),
        reinterpret_cast<hiprtApiFunction*>(&st->backward_func), nullptr, true));

    hiprtFuncDataSet fds{};
    HIPRT_CHECK(hiprtCreateFuncTable(st->context, 1, 1, st->func_table));
    HIPRT_CHECK(hiprtSetFuncTable(st->context, st->func_table, 0, 0, fds));
}


OptiXStateWrapper::~OptiXStateWrapper()
{
    if (!optixState) return;
    OptiXState* st = optixState;
    if (st->func_table)
        HIPRT_CHECK(hiprtDestroyFuncTable(st->context, st->func_table));
    if (st->has_geom && st->geom)
        HIPRT_CHECK(hiprtDestroyGeometry(st->context, st->geom));
    if (st->d_geom_temp)
        oroFree((oroDeviceptr)st->d_geom_temp);
    if (st->context)
        HIPRT_CHECK(hiprtDestroyContext(st->context));
    delete optixState;
    optixState = nullptr;
}


void hiprtTracerBuildGeometry(
    OptiXStateWrapper& wrapper,
    const float* d_vertices, int num_vertices,
    const int* d_triangles, int num_triangles,
    unsigned int rebuild, void* hip_stream)
{
    OptiXState* st = wrapper.optixState;
    oroStream stream = reinterpret_cast<oroStream>(hip_stream);

    hiprtTriangleMeshPrimitive mesh{};
    mesh.vertices = const_cast<void*>(reinterpret_cast<const void*>(d_vertices));
    mesh.vertexCount = (uint32_t)num_vertices;
    mesh.vertexStride = 3 * sizeof(float);
    mesh.triangleIndices = const_cast<void*>(reinterpret_cast<const void*>(d_triangles));
    mesh.triangleCount = (uint32_t)num_triangles;
    mesh.triangleStride = 3 * sizeof(uint32_t);

    hiprtGeometryBuildInput bi{};
    bi.type = hiprtPrimitiveTypeTriangleMesh;
    bi.primitive.triangleMesh = mesh;
    bi.geomType = 0;

    hiprtBuildOptions bo{};
    bo.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    size_t tempSize = 0;
    HIPRT_CHECK(hiprtGetGeometryBuildTemporaryBufferSize(st->context, bi, bo, tempSize));

    if (tempSize > st->geom_temp_size) {
        if (st->d_geom_temp) oroFree((oroDeviceptr)st->d_geom_temp);
        st->d_geom_temp = nullptr;
        if (tempSize) ORO_CHECK(oroMalloc((oroDeviceptr*)&st->d_geom_temp, tempSize));
        st->geom_temp_size = tempSize;
    }

    hiprtBuildOperation op = hiprtBuildOperationBuild;
    if (rebuild > 0 || !st->has_geom) {
        if (st->has_geom && st->geom) {
            HIPRT_CHECK(hiprtDestroyGeometry(st->context, st->geom));
            st->geom = nullptr;
            st->has_geom = false;
        }
        HIPRT_CHECK(hiprtCreateGeometry(st->context, bi, bo, st->geom));
        op = hiprtBuildOperationBuild;
    } else {
        op = hiprtBuildOperationUpdate;
    }

    HIPRT_CHECK(hiprtBuildGeometry(st->context, op, bi, bo, st->d_geom_temp, stream, st->geom));
    st->has_geom = true;
}


void hiprtTracerLaunch(
    const OptiXStateWrapper& wrapper, int which_kernel,
    const void* h_params, unsigned long params_size,
    int H, int W, void* hip_stream)
{
    OptiXState* st = wrapper.optixState;
    oroStream stream = reinterpret_cast<oroStream>(hip_stream);
    oroFunction func = which_kernel == 0 ? st->forward_func : st->backward_func;

    void* d_params = nullptr;
    ORO_CHECK(oroMalloc((oroDeviceptr*)&d_params, params_size));
    ORO_CHECK(oroMemcpyHtoDAsync((oroDeviceptr)d_params, const_cast<void*>(h_params), params_size, stream));

    hiprtGeometry geom = st->geom;
    hiprtFuncTable ftable = st->func_table;
    void* args[] = { &d_params, &geom, &ftable };

    int gx = (H + TRACE_BLOCK_X - 1) / TRACE_BLOCK_X;
    int gy = (W + TRACE_BLOCK_Y - 1) / TRACE_BLOCK_Y;
    oroError le = oroModuleLaunchKernel(func, gx, gy, 1, TRACE_BLOCK_X, TRACE_BLOCK_Y, 1,
                                    0, stream, args, nullptr);
    if (le != oroSuccess) {
        const char* m = nullptr;
        oroGetErrorString(le, &m);
        std::cerr << "ERROR: trace kernel launch failed: " << (m ? m : "unknown")
                  << " (" << (int)le << ")\n";
    }
    oroError se = oroStreamSynchronize(stream);
    if (se != oroSuccess) {
        const char* m = nullptr;
        oroGetErrorString(se, &m);
        std::cerr << "ERROR: trace kernel execution failed: " << (m ? m : "unknown")
                  << " (" << (int)se << ")\n";
    }

    oroFree((oroDeviceptr)d_params);
}
