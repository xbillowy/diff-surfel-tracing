/**
 * @file kernels.h
 * @brief HIPRT device kernels for the differentiable 2DGS surfel tracer.
 *
 * This is the ROCm/HIP reimplementation of the OptiX optix_tracer device code
 * (forward.cu + backward.cu). It is JIT-compiled at runtime by HIPRT (via
 * Orochi -> hiprtc) as a single translation unit and contains:
 *   - the forward path-tracing kernel  forward_kernel  (was __raygen__ot fwd)
 *   - the backward path-tracing kernel backward_kernel (was __raygen__ot bwd)
 *   - the hit-collection filter functor surfelFilter   (was __anyhit__ot)
 *
 * OptiX -> HIPRT mapping:
 *   optixTrace(handle, o, d, ...)         -> hiprtGeomTraversalAnyHit tr(geom,
 *                                            ray, hint, &payload, funcTable);
 *                                            tr.getNextHit();  // drives filter
 *   __anyhit__ (record-and-ignore)        -> surfelFilter returning true (reject)
 *                                            so the AnyHit traversal enumerates
 *                                            EVERY intersection along the ray.
 *   optixGetRayTmax()/optixGetPrimitiveIndex() -> hit.t / hit.primID
 *   optixIgnoreIntersection()             -> return true from surfelFilter
 *   __constant__ Params params            -> Params passed as a kernel argument
 *   optixGetLaunchIndex()/Dimensions()    -> derived from blockIdx/threadIdx;
 *                                            the launch grid is (H, W) so the
 *                                            per-thread index is idx=(h, w, 0),
 *                                            dim=(H, W, 1), tidx = h*W + w,
 *                                            byte-identical to the OptiX (H,W,1)
 *                                            launch the original used.
 *
 * The volume-rendering, chunked-retrace, and gradient math are reused verbatim
 * from the OptiX sources; only the ray-traversal calls and launch-index/param
 * access differ.
 */

#include <hip/hip_runtime.h>
#include <hiprt/hiprt_device.h>

#include "params.h"
#include "auxiliary.h"


// ---------------------------------------------------------------------------
// Spherical-harmonics -> RGB (forward), identical to optix_tracer/forward.cu.
// ---------------------------------------------------------------------------
__device__ float3 computeColorFromSH(int deg, const float3* sh, const float3& dir)
{
    float3 result = SH_C0 * sh[0];
    if (deg > 0)
    {
        float x = dir.x, y = dir.y, z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];
            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;
    return max(result, 0.0f);
}


// ---------------------------------------------------------------------------
// SH -> RGB (backward), identical to optix_tracer/backward.cu.
// ---------------------------------------------------------------------------
__device__ float3 computeColorFromSH(int deg, const float3* sh, const float3& dir, float* clamped)
{
    float3 result = SH_C0 * sh[0];
    if (deg > 0)
    {
        float x = dir.x, y = dir.y, z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];
            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;
    clamped[0] = (result.x < 0);
    clamped[1] = (result.y < 0);
    clamped[2] = (result.z < 0);
    return max(result, 0.0f);
}

__device__ __noinline__ void computeColorFromSHBackward(int deg, const float3* sh, const float3& dir, const float* clamped, const float* dL_dcolor, float3* dL_dsh, float3& dL_ddir)
{
    float3 dL_dRGB = make_float3(dL_dcolor[0], dL_dcolor[1], dL_dcolor[2]);
    dL_dRGB.x *= clamped[0] ? 0 : 1;
    dL_dRGB.y *= clamped[1] ? 0 : 1;
    dL_dRGB.z *= clamped[2] ? 0 : 1;

    float3 dRGBdx = make_float3(0, 0, 0);
    float3 dRGBdy = make_float3(0, 0, 0);
    float3 dRGBdz = make_float3(0, 0, 0);
    float x = dir.x, y = dir.y, z = dir.z;

    float dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;
    if (deg > 0)
    {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

            if (deg > 2)
            {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9] = dRGBdsh9 * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                dRGBdx += (
                    SH_C3[0] * sh[9] * 3.f * 2.f * xy +
                    SH_C3[1] * sh[10] * yz +
                    SH_C3[2] * sh[11] * -2.f * xy +
                    SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                    SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                    SH_C3[5] * sh[14] * 2.f * xz +
                    SH_C3[6] * sh[15] * 3.f * (xx - yy));

                dRGBdy += (
                    SH_C3[0] * sh[9] * 3.f * (xx - yy) +
                    SH_C3[1] * sh[10] * xz +
                    SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
                    SH_C3[3] * sh[12] * -3.f * 2.f * yz +
                    SH_C3[4] * sh[13] * -2.f * xy +
                    SH_C3[5] * sh[14] * -2.f * yz +
                    SH_C3[6] * sh[15] * -3.f * 2.f * xy);

                dRGBdz += (
                    SH_C3[1] * sh[10] * xy +
                    SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                    SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                    SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                    SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }
    dL_ddir = make_float3(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));
}


// ---------------------------------------------------------------------------
// World<->splat transforms (forward + backward), identical to OptiX sources.
// ---------------------------------------------------------------------------
__device__ __noinline__ void compute_transmat_uv(
    const float3 p_orig, const float2 scale, float mod, const float4 rot,
    const float3 xyz, float4* world2splat, float3& normal, float2& uv)
{
    float3 R[3];
    quat_to_rotmat_transpose(rot, R);
    float3 T = matmul33x3(R, p_orig);

    world2splat[0] = make_float4(R[0].x, R[0].y, R[0].z, -T.x);
    world2splat[1] = make_float4(R[1].x, R[1].y, R[1].z, -T.y);
    world2splat[2] = make_float4(R[2].x, R[2].y, R[2].z, -T.z);
    world2splat[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    normal = make_float3(R[2].x, R[2].y, R[2].z);

    float4 uv1 = matmul44x4(world2splat, make_float4(xyz.x, xyz.y, xyz.z, 1.0f));
    uv = make_float2(uv1.x / scale.x, uv1.y / scale.y);
}

// Forward-named alias so the backward path's call sites stay byte-identical
// to optix_tracer/backward.cu (which named the forward transform *_forward).
__device__ inline void compute_transmat_uv_forward(
    const float3 p_orig, const float2 scale, float mod, const float4 rot,
    const float3 xyz, float4* world2splat, float3& normal, float2& uv)
{
    compute_transmat_uv(p_orig, scale, mod, rot, xyz, world2splat, normal, uv);
}


__device__ __noinline__ void compute_transmat_uv_backward(
	const float3 p_orig,
	const float2 scale, 
	float mod,
	const float4 rot,
    const float3 xyz,
	const float4* world2splat,
	const float normal_sign,
    const float2 uv,
	const float3 ray_o,
	const float3 ray_d,
	const float dpt,
	const float3 v1,
	const float3 v2,
	const float3 v3,
	const float3 h1,
	const float3 h2,
	const float3 h3,
	const float G,
	const float power_clamped,
	const float3 dL_dN,
	const float dL_dD,
	const float dL_dG,
	float3& dL_dray_o,
	float3& dL_dray_d,
	float2& dL_dscale,
	float4& dL_drot,
	float3& dL_dmean3D
)
{
	// Compute the gradient w.r.t. the uv
	float2 dL_duv = dL_dG * power_clamped * -G * uv;

	float3 dL_dR[3];
	// Compute the gradient w.r.t. the transposed rotation matrix
	dL_dR[0] = dL_duv.x * (xyz - p_orig) / scale.x;
	dL_dR[1] = dL_duv.y * (xyz - p_orig) / scale.y;
	dL_dR[2] = dL_dN * normal_sign;

	// Update the gradient w.r.t. the scale
	dL_dscale = dL_dG * power_clamped * (G * uv * uv / scale);

	// Update the gradient w.r.t. the mean3D
	float3 dG_dmean3D = G * (to_float3(world2splat[0]) * uv.x / scale.x + to_float3(world2splat[1]) * uv.y / scale.y);
	dL_dmean3D = dL_dG * power_clamped * dG_dmean3D;

	// Compute the gradient flow through the ray-triangle intersection
	float3 dL_dxyz = make_float3(
		dL_duv.x / scale.x * world2splat[0].x + dL_duv.y / scale.y * world2splat[1].x,
		dL_duv.x / scale.x * world2splat[0].y + dL_duv.y / scale.y * world2splat[1].y,
		dL_duv.x / scale.x * world2splat[0].z + dL_duv.y / scale.y * world2splat[1].z
	);
	float dL_dd = dL_dD + dot(dL_dxyz, ray_d);
	// Compute the gradient w.r.t. the triangle vertices
	// Define some useful middle variables
	float3 n = cross(v2 - v1, v3 - v1);
	float3 c = v1 - ray_o;
	// Numerator and denominator aliases
	float p = dot(n, c);
	float q = dot(n, ray_d);
	// Chain rule for the gradient
	float3 dL_dn = (c - p / q * ray_d) / q;
	float3 dL_dv1 = dL_dd * cross(v2 - v3, dL_dn) + dL_dd * n / q;
	float3 dL_dv2 = dL_dd * cross(v3 - v1, dL_dn);
	float3 dL_dv3 = dL_dd * cross(v1 - v2, dL_dn);

	// Update the gradient w.r.t. the ray origin and direction
	dL_dray_o += dL_dxyz + dL_dd * -n / q;
	dL_dray_d += dL_dxyz * dpt + dL_dd * -p / (q * q) * n;

	// Update the gradient w.r.t. the transposed rotation matrix R
	dL_dR[0].x += scale.x * (h1.x * dL_dv1.x + h2.x * dL_dv2.x + h3.x * dL_dv3.x);
	dL_dR[0].y += scale.x * (h1.x * dL_dv1.y + h2.x * dL_dv2.y + h3.x * dL_dv3.y);
	dL_dR[0].z += scale.x * (h1.x * dL_dv1.z + h2.x * dL_dv2.z + h3.x * dL_dv3.z);
	dL_dR[1].x += scale.y * (h1.y * dL_dv1.x + h2.y * dL_dv2.x + h3.y * dL_dv3.x);
	dL_dR[1].y += scale.y * (h1.y * dL_dv1.y + h2.y * dL_dv2.y + h3.y * dL_dv3.y);
	dL_dR[1].z += scale.y * (h1.y * dL_dv1.z + h2.y * dL_dv2.z + h3.y * dL_dv3.z);
	// Update gradient w.r.t. rotation
	dL_drot = quat_to_rotmat_vjp(rot, dL_dR);

	// Update the gradient w.r.t. the scale
	dL_dscale.x += dot(to_float3(world2splat[0]), h1.x * dL_dv1 + h2.x * dL_dv2 + h3.x * dL_dv3);
	dL_dscale.y += dot(to_float3(world2splat[1]), h1.y * dL_dv1 + h2.y * dL_dv2 + h3.y * dL_dv3);

	// Update the gradient w.r.t. the mean3D
	dL_dmean3D += dL_dv1 + dL_dv2 + dL_dv3;
}


// ---------------------------------------------------------------------------
// Hit-collection filter functor (the OptiX __anyhit__ot, both fwd and bwd).
//
// Records the intersection into the per-ray RayPayload chunk buffer using the
// exact t-sorted ascending insertion of the OptiX anyhit, then returns true to
// REJECT the candidate so the HIPRT AnyHit traversal continues and visits every
// triangle along the ray (validated on gfx90a: AnyHit + always-reject filter
// enumerates all hits, t-sorted). HIPRT generates the `filterFunc` dispatcher
// that calls this by the name passed in funcNameSet.filterFuncName.
// ---------------------------------------------------------------------------
__device__ bool surfelFilter(
    const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit)
{
    RayPayload& p = *reinterpret_cast<RayPayload*>(payload);

    float tmx = hit.t;
    unsigned int idx = hit.primID;

    if (tmx < p.buffer[CHUNK_SIZE - 1].tmx)
    {
        p.cnt += 1;

        float tmp_tmx;
        float cur_tmx = tmx;
        unsigned int tmp_idx;
        unsigned int cur_idx = idx;

        for (int i = 0; i < CHUNK_SIZE; ++i)
        {
            if (p.buffer[i].tmx > cur_tmx)
            {
                tmp_tmx = p.buffer[i].tmx;
                tmp_idx = p.buffer[i].idx;
                p.buffer[i].tmx = cur_tmx;
                p.buffer[i].idx = cur_idx;
                cur_tmx = tmp_tmx;
                cur_idx = tmp_idx;
            }
        }
    }

    return true;  // reject -> continue traversal (== optixIgnoreIntersection)
}


// One traversal that collects up to CHUNK_SIZE closest hits into the payload.
// Replaces the OptiX traceStep()/optixTrace(): constructs an AnyHit traversal
// with the surfel filter and runs it to completion (every hit is rejected, so
// getNextHit returns invalid; the side effect is the filled payload buffer).
// Runs the AnyHit traversal (drives surfelFilter over every hit) and RETURNS the
// resulting chunk count. The count is read here, inside this low-register-pressure
// __noinline__ frame, right after the traversal -- NOT in the caller. In a heavy
// caller frame (the forward/backward shading), the hiprtc optimizer otherwise
// mis-schedules the read of the filter-updated payload relative to this opaque
// call and sees a stale (zero) count; returning it from here makes the caller use
// a fresh value. The chunk buffer is global (params.chunk_buffer), so its filter
// writes are always visible; this return-value contract closes the count read too.
__device__ __noinline__ unsigned int traceStep(
    hiprtGeometry geom, hiprtFuncTable funcTable,
    const float3& ray_o, const float3& ray_d, RayPayload& payload)
{
    hiprtRay ray;
    ray.origin = ray_o;
    ray.direction = ray_d;
    ray.minT = 0.0f;
    ray.maxT = 1e16f;

    hiprtGeomTraversalAnyHit tr(geom, ray, hiprtTraversalHintDefault, &payload, funcTable);
    hiprtHit h = tr.getNextHit();
    (void)h;
    // Make the filter's global chunk-buffer writes visible to the caller's
    // subsequent reads (the hiprtc optimizer otherwise reorders/caches the
    // post-traversal buffer reads in a heavy caller frame).
    __threadfence();
    return payload.cnt;
}


// ===========================================================================
// FORWARD path tracing (was optix_tracer/forward.cu traceRay/tracePath).
// ===========================================================================
__device__ void traceRay_fwd(
    hiprtGeometry geom, hiprtFuncTable funcTable, const Params& params, const uint3 idx, const uint3 dim,
    const float3& ray_o, const float3& ray_d,
    const float min_depth, const float max_depth, const float T_threshold,
    const int trace_depth,
    float* C, float& D, float& A, float3& N,
    float& dist, float& M1, float& M2, float* O, float& T, float3& E)
{
    uint32_t tidx = idx.x * dim.y + idx.y;

    float3 ray_ot = ray_o;
    float3 ray_dt = ray_d;

    // Per-ray chunk buffer lives in GLOBAL scratch (params.chunk_buffer), not a
    // kernel stack array: the hit-collection filter writes through this pointer,
    // and above a register-pressure threshold the hiprtc-compiled traversal's
    // writes to a stack-payload buffer become invisible. Global is pressure-immune.
    RayPayload payload;
    IntersectionInfo* buffer = params.chunk_buffer + (size_t)tidx * CHUNK_SIZE;
    for (int i = 0; i < CHUNK_SIZE; i++) buffer[i].tmx = max_depth;
    payload.buffer = buffer;
    payload.dpt = 0.0f;
    payload.cnt = 0;

    int last_gidx = -1;
    int contributor = 0;
    float T_prev = 1.0f;
    float T_next = 1.0f;
    float dpt = 0.0f;
    float rho3d = 0.0f;
    float4 world2splat[4];
    float3 xyz;
    float3 normal;
    float2 uv;
    float3 result;

    while (1)
    {
        payload.cnt = traceStep(geom, funcTable, ray_ot, ray_dt, payload);

        for (int i = 0; i < CHUNK_SIZE; i++)
        {
            if (i >= payload.cnt)
                break;

            int pidx = payload.buffer[i].idx;
            int gidx = pidx / 2;
            if (gidx == last_gidx)
                continue;

            dpt = payload.buffer[i].tmx + payload.dpt;
            xyz = ray_o + dpt * ray_d;

            payload.buffer[i].tmx = max_depth;
            payload.buffer[i].idx = 0;

            compute_transmat_uv(params.means3D[gidx], params.scales[gidx],
                                params.scale_modifier, params.rotations[gidx],
                                xyz, world2splat, normal, uv);
            rho3d = dot(uv, uv);

#if DUAL_VISIABLE
            float3 dir = ray_d;
            float cos = -sumf3(dir * normal);
            if (cos == 0) continue;
            normal = cos > 0 ? normal : -normal;
#endif

            if (dpt < min_depth)
                continue;

            float power = -0.5f * rho3d;
            if (power > 0.0f)
                continue;

            float alpha = min(0.99f, params.opacities[gidx] * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;
            T_next = T_prev * (1 - alpha);
            if (T_next < T_threshold)
                break;

            contributor++;
            if (contributor > MAX_INTERSECTION)
                break;
            last_gidx = gidx;

            float w = alpha * T_prev;

            if (params.colors_precomp == nullptr)
            {
                result = computeColorFromSH(params.D, &params.shs[gidx * params.M], ray_d);
                C[0] += w * result.x;
                C[1] += w * result.y;
                C[2] += w * result.z;
            }
            else
            {
                for (int ch = 0; ch < NUM_CHANNELS; ch++)
                    C[ch] += w * params.colors_precomp[ch + NUM_CHANNELS * gidx];
            }
            if (params.others_precomp != nullptr)
            {
                for (int ch = 0; ch < AUX_CHANNELS; ch++)
                    O[ch] += w * params.others_precomp[ch + AUX_CHANNELS * gidx];
            }
            D += w * dpt;
            N += w * normal;
            // TODO (xbillowy): maybe add distortion computation

            T_prev = T_next;

            if (params.training)
            {
                atomicAdd(&(params.a_weights[gidx]), w);
            }
        }

        if (T_next < T_threshold || payload.cnt < CHUNK_SIZE || contributor > MAX_INTERSECTION)
            break;

        payload.dpt = dpt + STEP_EPSILON;  // avoid self-intersection
        payload.cnt = 0;
        ray_ot = ray_o + payload.dpt * ray_d;
    }

    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        C[ch] += T_prev * params.background[ch];
    A = 1 - T_prev;
    T = T_prev;
    E = ray_o + D * ray_d;
}


__device__ void tracePath_fwd(
    hiprtGeometry geom, hiprtFuncTable funcTable, const Params& params, const uint3 idx, const uint3 dim,
    const float3& ray_o, const float3& ray_d, const float max_trace_depth,
    float* out_rgb, float& out_dpt, float& out_acc, float3& out_norm, float3& out_dist,
    float* out_aux, float* mid_val)
{
    float3 ray_ot = ray_o;
    float3 ray_dt = ray_d;

    float s_prod = 1.0f;

    for (int i = 0; i < max_trace_depth + 1; i++)
    {
        float C[NUM_CHANNELS] = {0.0f};
        float D = 0.0f;
        float A = 0.0f;
        float3 N = make_float3(0.0f, 0.0f, 0.0f);
        float dist = 0.0f;
        float M1 = 0.0f;
        float M2 = 0.0f;
        float O[AUX_CHANNELS] = {0.0f};
        float T = 1.0f;
        float3 E = make_float3(0.0f, 0.0f, 0.0f);

        float min_depth = (i == 0 && params.start_from_first) ? near_n : (i == 0  && !params.start_from_first) ? 0.0f : START_OFFSET;
        float max_depth = DEPTH_INFINTY;
        float T_threshold = (i == 0 && params.start_from_first) ? 0.0001f : 0.0001f;

        traceRay_fwd(geom, funcTable, params, idx, dim,
                     ray_ot, ray_dt, min_depth, max_depth, T_threshold, i,
                     C, D, A, N, dist, M1, M2, O, T, E);

        if (i == 0)
        {
            out_dpt = D;
            out_acc = A;
            out_norm = N;
            out_dist = make_float3(dist, M1, M2);
            for (int ch = 0; ch < AUX_CHANNELS; ch++)
                out_aux[ch] = O[ch];
        }
        float f_this = (i == max_trace_depth) ? 1.0f : 1 - O[SPECULAR_OFFSET];
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            out_rgb[ch] += C[ch] * f_this * s_prod;
        s_prod *= O[SPECULAR_OFFSET];

        mid_val[0 + RAYO_MID_OFFSET + i * MID_CHANNELS] = ray_ot.x;
        mid_val[1 + RAYO_MID_OFFSET + i * MID_CHANNELS] = ray_ot.y;
        mid_val[2 + RAYO_MID_OFFSET + i * MID_CHANNELS] = ray_ot.z;
        mid_val[0 + RAYD_MID_OFFSET + i * MID_CHANNELS] = ray_dt.x;
        mid_val[1 + RAYD_MID_OFFSET + i * MID_CHANNELS] = ray_dt.y;
        mid_val[2 + RAYD_MID_OFFSET + i * MID_CHANNELS] = ray_dt.z;
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            mid_val[ch + RGB_MID_OFFSET + i * MID_CHANNELS] = C[ch];
        mid_val[DPT_MID_OFFSET + i * MID_CHANNELS] = D;
        mid_val[ACC_MID_OFFSET + i * MID_CHANNELS] = A;
        mid_val[0 + NORM_MID_OFFSET + i * MID_CHANNELS] = N.x;
        mid_val[1 + NORM_MID_OFFSET + i * MID_CHANNELS] = N.y;
        mid_val[2 + NORM_MID_OFFSET + i * MID_CHANNELS] = N.z;
        for (int ch = 0; ch < AUX_CHANNELS; ch++)
            mid_val[ch + AUX_MID_OFFSET + i * MID_CHANNELS] = O[ch];

        if (length(N) == 0.0f || O[SPECULAR_OFFSET] <= params.specular_threshold)
            break;

        float3 n = normalize(N);
        ray_dt = ray_dt - 2 * dot(n, ray_dt) * n;
        ray_ot = E + START_OFFSET * ray_dt;
    }
}


extern "C" __global__ void __launch_bounds__(64) forward_kernel(const Params* params_ptr, hiprtGeometry geom, hiprtFuncTable funcTable)
{
    const Params& params = *params_ptr;
    // OptiX launched (H, W, 1); reproduce idx=(h, w, 0), dim=(H, W, 1).
    const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t w = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= (uint32_t)params.H || w >= (uint32_t)params.W) return;
    const uint3 idx = make_uint3(h, w, 0);
    const uint3 dim = make_uint3(params.H, params.W, 1);
    uint32_t tidx = idx.x * dim.y + idx.y;

    float3 ray_o = params.ray_o[tidx];
    float3 ray_d = params.ray_d[tidx];

    float out_rgb[NUM_CHANNELS] = {0.0f};
    float out_dpt = 0.0f;
    float out_acc = 0.0f;
    float3 out_norm = make_float3(0.0f, 0.0f, 0.0f);
    float3 out_dist = make_float3(0.0f, 0.0f, 0.0f);
    float out_aux[AUX_CHANNELS] = {0.0f};
    float mid_val[MID_CHANNELS * (MAX_TRACE_DEPTH + 1)];

    tracePath_fwd(geom, funcTable, params, idx, dim,
                  ray_o, ray_d, params.max_trace_depth,
                  out_rgb, out_dpt, out_acc, out_norm, out_dist, out_aux, mid_val);

    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        params.out_rgb[tidx * NUM_CHANNELS + ch] = out_rgb[ch];
    params.out_dpt[tidx] = out_dpt;
    params.out_acc[tidx] = out_acc;
    params.out_norm[tidx] = out_norm;
    params.out_dist[tidx] = out_dist;
    for (int ch = 0; ch < AUX_CHANNELS; ch++)
        params.out_aux[tidx * AUX_CHANNELS + ch] = out_aux[ch];
    for (int i = 0; i < params.max_trace_depth + 1; i++)
    {
        for (int ch = 0; ch < MID_CHANNELS; ch++)
            params.mid_val[ch + i * MID_CHANNELS + (MAX_TRACE_DEPTH + 1) * MID_CHANNELS * tidx] =
                mid_val[ch + i * MID_CHANNELS];
    }
}


// ===========================================================================
// BACKWARD path tracing (was optix_tracer/backward.cu traceRay/tracePath).
// Differentiable backward traversal: re-runs the same chunked traversal and
// accumulates gradients into the per-Gaussian dL_d* buffers via atomicAdd.
// The gradient math is reused verbatim from the OptiX source; only the
// traversal call and launch-index/param access differ.
// ===========================================================================
__device__ void traceRay_bwd(
    hiprtGeometry geom, hiprtFuncTable funcTable, const Params& params, const uint3 idx, const uint3 dim,
    const float3& ray_o, const float3& ray_d,
    const float min_depth, const float max_depth, const float T_threshold,
    const int trace_depth,
    const float* out_rgb, const float& out_dpt, const float& out_acc,
    const float3& out_norm, const float3& out_dist, const float* out_aux,
    const float* dL_drgb, const float dL_ddpt, const float dL_dacc,
    const float3 dL_dnorm, const float dL_ddist, const float* dL_daux,
    float* C, float* clamped, float& D, float& W, float3& N,
    float& dist, float& M1, float& M2, float* O,
    float3& dL_dray_o, float3& dL_dray_d)
{
    uint32_t tidx = idx.x * dim.y + idx.y;

    float3 ray_ot = ray_o;
    float3 ray_dt = ray_d;

    // Global per-ray chunk scratch (see traceRay_fwd / params.h for why it must
    // not be a stack array under the heavy backward register pressure).
    RayPayload payload;
    IntersectionInfo* buffer = params.chunk_buffer + (size_t)tidx * CHUNK_SIZE;
    for (int i = 0; i < CHUNK_SIZE; i++) buffer[i].tmx = max_depth;
    payload.buffer = buffer;
    payload.dpt = 0.0f;
    payload.cnt = 0;

    int last_gidx = -1;
    int contributor = 0;
    float T_prev = 1.0f;
    float T_next = 1.0f;
    float c[NUM_CHANNELS] = {0.0f};
    float dpt = 0.0f;
    float rho3d = 0.0f;
    float4 world2splat[4];
    float3 xyz;
    float3 normal;
    float normal_sign = 1.0f;
    float2 uv;
    float3 result;

    float cutoff;
#if TIGHTBBOX
    // The effective extent maybe depend on the opacity of Gaussian
    cutoff = sqrtf(max(9.f + 2.f * logf(params.opacities[gidx]), 0.000001));
#else
    cutoff = 3.0f;
#endif

    float T_final = 1.0f - out_acc;
    float dL_dcolor[NUM_CHANNELS];
    float3 dL_dsh[MAX_SH_COEFFS];
    float2 dL_dscale;
    float4 dL_drot;
    float3 dL_dmean3D;

    while (1)
    {
        payload.cnt = traceStep(geom, funcTable, ray_ot, ray_dt, payload);

        for (int i = 0; i < CHUNK_SIZE; i++)
        {
            if (i >= payload.cnt)
                break;

            int pidx = payload.buffer[i].idx;
            int gidx = pidx / 2;
            if (gidx == last_gidx)
                continue;

            dpt = payload.buffer[i].tmx + payload.dpt;
            xyz = ray_o + dpt * ray_d;

            payload.buffer[i].tmx = max_depth;
            payload.buffer[i].idx = 0;

            compute_transmat_uv_forward(params.means3D[gidx], params.scales[gidx],
                                        params.scale_modifier, params.rotations[gidx],
                                        xyz, world2splat, normal, uv);
            rho3d = dot(uv, uv);

#if DUAL_VISIABLE
            float3 dir = ray_d;
            float cos = -sumf3(dir * normal);
            if (cos == 0) continue;
            normal_sign = cos > 0 ? 1.0f : -1.0f;
            normal = normal_sign * normal;
#endif

            if (dpt < min_depth)
                continue;

            float power = -0.5f * rho3d;
            if (power > 0.0f)
                continue;

            const float G = exp(power);
            float alpha = min(0.99f, params.opacities[gidx] * G);
            if (alpha < 1.0f / 255.0f)
                continue;
            T_next = T_prev * (1 - alpha);
            if (T_next < T_threshold)
                break;

            contributor++;
            if (contributor > MAX_INTERSECTION)
                break;
            last_gidx = gidx;

            float w = alpha * T_prev;

            if (params.colors_precomp == nullptr)
            {
                result = computeColorFromSH(params.D, &params.shs[gidx * params.M], ray_d, clamped);
                C[0] += w * result.x;
                C[1] += w * result.y;
                C[2] += w * result.z;
                c[0] = result.x;
                c[1] = result.y;
                c[2] = result.z;
            }
            else
            {
                for (int ch = 0; ch < NUM_CHANNELS; ch++)
                {
                    C[ch] += w * params.colors_precomp[ch + NUM_CHANNELS * gidx];
                    c[ch] = params.colors_precomp[ch + NUM_CHANNELS * gidx];
                }
            }
            if (params.others_precomp != nullptr)
            {
                for (int ch = 0; ch < AUX_CHANNELS; ch++)
                    O[ch] += w * params.others_precomp[ch + AUX_CHANNELS * gidx];
            }
            D += w * dpt;
            N += w * normal;
            // TODO (xbillowy): maybe add distortion computation

            // Backward pass: dL_dalpha = dL_dF * (T_{i-1} * f_i + (F - F_i) / (1 - alpha_i))
            float dL_dalpha = 0.0f;
            float numerator = 1.0f - alpha;  // better accuracy?

            for (int ch = 0; ch < NUM_CHANNELS; ch++)
            {
                const float channel = c[ch];
                const float dL_dchannel = dL_drgb[ch];
                dL_dcolor[ch] = dL_dchannel * w;
                atomicAdd(&(params.dL_dcolors[ch + NUM_CHANNELS * gidx]), dL_dchannel * w);
                dL_dalpha += dL_dchannel * (T_prev * channel - (out_rgb[ch] - C[ch]) / numerator);
                dL_dalpha += dL_dchannel * params.background[ch] * (-T_final / numerator);
            }
            if (params.others_precomp != nullptr)
            {
                for (int ch = 0; ch < AUX_CHANNELS; ch++)
                {
                    const float channel = params.others_precomp[ch + AUX_CHANNELS * gidx];
                    const float dL_dchannel = dL_daux[ch];
                    atomicAdd(&(params.dL_dothers[ch + AUX_CHANNELS * gidx]), dL_dchannel * w);
                    dL_dalpha += dL_dchannel * (T_prev * channel - (out_aux[ch] - O[ch]) / numerator);
                }
            }
            float dL_dD = dL_ddpt * w;
            dL_dalpha += dL_ddpt * (T_prev * dpt - (out_dpt - D) / numerator);
            float3 dL_dN = dL_dnorm * w;
            dL_dalpha += sumf3(dL_dnorm * (T_prev * normal - (out_norm - N) / numerator));
            dL_dalpha += dL_dacc * (T_prev * 1.f - (out_acc - W) / numerator);
            // TODO (xbillowy): implement the distortion loss gradient

            const float dL_dG = dL_dalpha * params.opacities[gidx];
            float power_clamped = (params.opacities[gidx] * G) > 0.99f ? 0.0f : 1.0f;

            float dL_dopacity = dL_dalpha * G * power_clamped;
            atomicAdd(&(params.dL_dopacities[gidx]), dL_dopacity);

            float3 v1, v2, v3, h1, h2, h3;
            if (pidx % 2 == 0)
            {
                v1 = params.vertices[gidx * 4 + 0];
                v2 = params.vertices[gidx * 4 + 1];
                v3 = params.vertices[gidx * 4 + 2];
                h1 = make_float3(-1.0f,  1.0f, 1.0f) * cutoff;
                h2 = make_float3(-1.0f, -1.0f, 1.0f) * cutoff;
                h3 = make_float3( 1.0f,  1.0f, 1.0f) * cutoff;
            }
            else
            {
                v1 = params.vertices[gidx * 4 + 1];
                v2 = params.vertices[gidx * 4 + 2];
                v3 = params.vertices[gidx * 4 + 3];
                h1 = make_float3(-1.0f, -1.0f, 1.0f) * cutoff;
                h2 = make_float3( 1.0f,  1.0f, 1.0f) * cutoff;
                h3 = make_float3( 1.0f, -1.0f, 1.0f) * cutoff;
            }
            compute_transmat_uv_backward(params.means3D[gidx], params.scales[gidx],
                                         params.scale_modifier, params.rotations[gidx],
                                         xyz, world2splat, normal_sign, uv, ray_o, ray_d, dpt, v1, v2, v3, h1, h2, h3,
                                         G, power_clamped, dL_dN, dL_dD, dL_dG,
                                         dL_dray_o, dL_dray_d, dL_dscale, dL_drot, dL_dmean3D);

            atomicAdd(&(params.dL_dscales[gidx].x), dL_dscale.x);
            atomicAdd(&(params.dL_dscales[gidx].y), dL_dscale.y);
            atomicAdd(&(params.dL_drotations[gidx].x), dL_drot.x);
            atomicAdd(&(params.dL_drotations[gidx].y), dL_drot.y);
            atomicAdd(&(params.dL_drotations[gidx].z), dL_drot.z);
            atomicAdd(&(params.dL_drotations[gidx].w), dL_drot.w);
            atomicAdd(&(params.dL_dmeans3D[gidx].x), dL_dmean3D.x);
            atomicAdd(&(params.dL_dmeans3D[gidx].y), dL_dmean3D.y);
            atomicAdd(&(params.dL_dmeans3D[gidx].z), dL_dmean3D.z);

            // Update the accumulated gradients for densification
            // NOTE: scale the gradients by half depth to avoid far distance pruning
            atomicAdd(&(params.dL_dgrads3D[gidx].x), dL_dmean3D.x * 0.5f * dpt);
            atomicAdd(&(params.dL_dgrads3D[gidx].y), dL_dmean3D.y * 0.5f * dpt);
            atomicAdd(&(params.dL_dgrads3D[gidx].z), dL_dmean3D.z * 0.5f * dpt);

            if (params.colors_precomp == nullptr)
            {
                computeColorFromSHBackward(params.D, &params.shs[gidx * params.M], ray_d, clamped,
                                           dL_dcolor, dL_dsh, dL_dray_d);
                for (int j = 0; j < (params.D + 1) * (params.D + 1); j++)
                {
                    atomicAdd(&(params.dL_dshs[j + params.M * gidx].x), dL_dsh[j].x);
                    atomicAdd(&(params.dL_dshs[j + params.M * gidx].y), dL_dsh[j].y);
                    atomicAdd(&(params.dL_dshs[j + params.M * gidx].z), dL_dsh[j].z);
                }
            }

            T_prev = T_next;
        }

        if (T_next < T_threshold || payload.cnt < CHUNK_SIZE || contributor > MAX_INTERSECTION)
            break;

        payload.dpt = dpt + STEP_EPSILON;  // avoid self-intersection
        payload.cnt = 0;
        ray_ot = ray_o + payload.dpt * ray_d;
    }
}


__device__ void tracePath_bwd(
    hiprtGeometry geom, hiprtFuncTable funcTable, const Params& params, const uint3 idx, const uint3 dim,
    const float3& ray_o, const float3& ray_d, const float max_trace_depth,
    const float* out_rgb, const float& out_dpt, const float& out_acc,
    const float3& out_norm, const float3& out_dist, const float* out_aux, const float* mid_val,
    const float* dL_dout_rgb, const float dL_dout_dpt, const float dL_dout_acc,
    const float3 dL_dout_norm, const float dL_dout_dist, const float* dL_dout_aux,
    float3& dL_dray_o, float3& dL_dray_d)
{
    float3 ray_ot;
    float3 ray_dt;

    float C_prev[NUM_CHANNELS];
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        C_prev[ch] = out_rgb[ch];
    float c_prev[NUM_CHANNELS] = {0.0f};
    float s_prod = 1.0f;
    for (int i = 0; i < max_trace_depth; i++)
    {
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
        {
            float c_curr = mid_val[ch + RGB_MID_OFFSET + i * MID_CHANNELS];
            C_prev[ch] = C_prev[ch] + s_prod * (c_prev[ch] - c_curr);
            c_prev[ch] = c_curr;
        }
        s_prod *= mid_val[SPECULAR_OFFSET + AUX_MID_OFFSET + i * MID_CHANNELS];
    }

    for (int i = max_trace_depth; i >= 0; i--)
    {
        ray_ot = make_float3(
            mid_val[0 + RAYO_MID_OFFSET + i * MID_CHANNELS],
            mid_val[1 + RAYO_MID_OFFSET + i * MID_CHANNELS],
            mid_val[2 + RAYO_MID_OFFSET + i * MID_CHANNELS]
        );
        ray_dt = make_float3(
            mid_val[0 + RAYD_MID_OFFSET + i * MID_CHANNELS],
            mid_val[1 + RAYD_MID_OFFSET + i * MID_CHANNELS],
            mid_val[2 + RAYD_MID_OFFSET + i * MID_CHANNELS]
        );
        float s_prev = (i > 0) ? mid_val[SPECULAR_OFFSET + AUX_MID_OFFSET + (i - 1) * MID_CHANNELS] : 1.0f;

        if (length(ray_dt) == 0.0f || s_prev <= params.specular_threshold)
            continue;

        float rgb[NUM_CHANNELS] = {0.0f};
        float dpt = 0.0f;
        float acc = 0.0f;
        float3 norm = make_float3(0.0f, 0.0f, 0.0f);
        float3 dist = make_float3(0.0f, 0.0f, 0.0f);
        float aux[AUX_CHANNELS] = {0.0f};
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            rgb[ch] = mid_val[ch + RGB_MID_OFFSET + i * MID_CHANNELS];
        dpt = mid_val[DPT_MID_OFFSET + i * MID_CHANNELS];
        acc = mid_val[ACC_MID_OFFSET + i * MID_CHANNELS];
        norm = make_float3(
            mid_val[0 + NORM_MID_OFFSET + i * MID_CHANNELS],
            mid_val[1 + NORM_MID_OFFSET + i * MID_CHANNELS],
            mid_val[2 + NORM_MID_OFFSET + i * MID_CHANNELS]
        );
        for (int ch = 0; ch < AUX_CHANNELS; ch++)
            aux[ch] = mid_val[ch + AUX_MID_OFFSET + i * MID_CHANNELS];

        float dL_drgb[NUM_CHANNELS] = {0.0f};
        float dL_ddpt = 0.0f;
        float dL_dacc = 0.0f;
        float3 dL_dnorm = make_float3(0.0f, 0.0f, 0.0f);
        float dL_ddist = 0.0f;
        float dL_daux[AUX_CHANNELS] = {0.0f};
        float f_this = (i == max_trace_depth) ? 1.0f : 1 - aux[SPECULAR_OFFSET];
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            dL_drgb[ch] = dL_dout_rgb[ch] * f_this * s_prod;
        s_prod /= s_prev;
        if (i != max_trace_depth)
        {
            for (int ch = 0; ch < NUM_CHANNELS; ch++)
            {
                dL_daux[SPECULAR_OFFSET] += dL_dout_rgb[ch] * (C_prev[ch] / aux[SPECULAR_OFFSET]);
                c_prev[ch] = (i > 0) ? mid_val[ch + RGB_MID_OFFSET + (i - 1) * MID_CHANNELS] : 0.0f;
                C_prev[ch] = C_prev[ch] - s_prod * (c_prev[ch] - rgb[ch]);
            }

            if (length(norm) != 0)
            {
                dL_dray_d = dL_dray_d + dL_dray_o * STEP_EPSILON;
                float3 n = normalize(norm);
                dL_ddpt += sumf3(dL_dray_o * ray_dt);
                dL_dnorm += dnormvdv(norm, ddotndndn(n, ray_dt, dL_dray_d * -2.0f));
                dL_dray_o = dL_dray_o;
                dL_dray_d = dL_dray_o * dpt + dL_dray_d + ddotndndd(n, ray_dt, dL_dray_d * -2.0f);
            }
        }
        if (i == 0)
        {
            dL_ddpt += dL_dout_dpt;
            dL_dacc += dL_dout_acc;
            dL_dnorm += dL_dout_norm;
            dL_ddist += dL_dout_dist;
            for (int ch = 0; ch < AUX_CHANNELS; ch++)
                dL_daux[ch] += dL_dout_aux[ch];
        }

        float C[NUM_CHANNELS] = {0.0f};
        float clamped[3] = {0.0f};
        float D = 0.0f;
        float W = 0.0f;
        float3 N = make_float3(0.0f, 0.0f, 0.0f);
        float DS = 0.0f;
        float M1 = 0.0f;
        float M2 = 0.0f;
        float O[AUX_CHANNELS] = {0.0f};

        float min_depth = (i == 0 && params.start_from_first) ? near_n : (i == 0  && !params.start_from_first) ? 0.0f : START_OFFSET;
        float max_depth = DEPTH_INFINTY;
        float T_threshold = (i == 0 && params.start_from_first) ? 0.0001f : 0.0001f;

        traceRay_bwd(geom, funcTable, params, idx, dim,
                     ray_ot, ray_dt, min_depth, max_depth, T_threshold, i,
                     rgb, dpt, acc, norm, dist, aux,
                     dL_drgb, dL_ddpt, dL_dacc, dL_dnorm, dL_ddist, dL_daux,
                     C, clamped, D, W, N, DS, M1, M2, O,
                     dL_dray_o, dL_dray_d);
    }
}


extern "C" __global__ void __launch_bounds__(64) backward_kernel(const Params* params_ptr, hiprtGeometry geom, hiprtFuncTable funcTable)
{
    const Params& params = *params_ptr;
    const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t w = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= (uint32_t)params.H || w >= (uint32_t)params.W) return;
    const uint3 idx = make_uint3(h, w, 0);
    const uint3 dim = make_uint3(params.H, params.W, 1);
    uint32_t tidx = idx.x * dim.y + idx.y;

    float3 ray_o = params.ray_o[tidx];
    float3 ray_d = params.ray_d[tidx];

    // Read the forward outputs and the saved per-bounce mid_val straight from
    // global memory (do NOT stage them into local stack arrays). Under the heavy
    // backward register pressure, hiprtc/comgr (ROCm 7.2.x) miscompiles reads of
    // large kernel-local arrays loaded from global -- the values come back stale
    // (NaN/garbage), which then poisons dL_dalpha (the out_rgb-C term) and every
    // gradient. Global pointers are pressure-immune, which is why the per-ray hit
    // chunk lives in params.chunk_buffer rather than on the stack.
    const float* out_rgb = params.out_rgb + NUM_CHANNELS * tidx;
    float out_dpt = params.out_dpt[tidx];
    float out_acc = params.out_acc[tidx];
    float3 out_norm = params.out_norm[tidx];
    float3 out_dist = params.out_dist[tidx];
    const float* out_aux = params.out_aux + AUX_CHANNELS * tidx;
    const float* mid_val = params.mid_val + MID_CHANNELS * (MAX_TRACE_DEPTH + 1) * tidx;

    const float* dL_drgb = params.dL_drgb + NUM_CHANNELS * tidx;
    float dL_ddpt = params.dL_ddpt[tidx];
    float dL_dacc = params.dL_dacc[tidx];
    float3 dL_dnorm = params.dL_dnorm[tidx];
    float dL_ddist = params.dL_ddist[tidx];
    const float* dL_daux = params.dL_daux + AUX_CHANNELS * tidx;

    float3 dL_dray_o = make_float3(0.0f, 0.0f, 0.0f);
    float3 dL_dray_d = make_float3(0.0f, 0.0f, 0.0f);

    tracePath_bwd(geom, funcTable, params, idx, dim,
                  ray_o, ray_d, params.max_trace_depth,
                  out_rgb, out_dpt, out_acc, out_norm, out_dist, out_aux, mid_val,
                  dL_drgb, dL_ddpt, dL_dacc, dL_dnorm, dL_ddist, dL_daux,
                  dL_dray_o, dL_dray_d);

    params.dL_dray_o[tidx] = dL_dray_o;
    params.dL_dray_d[tidx] = dL_dray_d;
}
