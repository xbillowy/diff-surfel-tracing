"""GPU validation harness for the diff-surfel-tracing HIP RT (ROCm) back end.

Self-contained: builds a small synthetic 2D-Gaussian scene, tessellates it into
surfel disks exactly the way the README's get_triangles() does, traces it, and
checks the forward image, the seven backward gradients this scene exercises,
finite-difference agreement, a reflected bounce, scenes large enough to reach
the multi-block sort in the acceleration-structure build, and a cold
runtime-compilation cache.

The tracer returns eleven gradients; ray_o and ray_d are not differentiated
here, and shs and cov3D_precomp are the alternative parameterization of the
colors and the scale and rotation pair this scene uses.

Run, from a checkout with the extension installed:
    HIP_VISIBLE_DEVICES=0 python3 example/validate_rocm.py
"""

import os
import shutil
import sys

import torch

import diff_surfel_tracing
from diff_surfel_tracing import SurfelTracer, SurfelTracingSettings

DEV = "cuda"
FAILURES = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def make_scene(P=64, seed=0):
    """A slab of randomly oriented surfels in front of the camera, plus one
    large, highly specular, near-planar surfel behind them to reflect off."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    means3D = torch.empty(P, 3)
    means3D[:, 0].uniform_(-0.6, 0.6, generator=g)
    means3D[:, 1].uniform_(-0.6, 0.6, generator=g)
    means3D[:, 2].uniform_(1.5, 3.5, generator=g)

    scales = torch.empty(P, 2).uniform_(0.06, 0.16, generator=g)
    rotations = torch.randn(P, 4, generator=g)
    rotations = rotations / rotations.norm(dim=-1, keepdim=True)
    opacities = torch.empty(P, 1).uniform_(0.35, 0.9, generator=g)
    colors = torch.empty(P, 3).uniform_(0.1, 0.9, generator=g)

    # others_precomp carries AUX_CHANNELS (config.h) values per surfel, and the
    # kernel indexes it with that stride, so it has to be exactly that wide.
    # Channel SPECULAR_OFFSET (0) is the specular weight that drives reflection.
    others = torch.empty(P, 2).uniform_(0.0, 0.05, generator=g)
    # A big, strongly specular mirror surfel at the back of the slab.
    means3D[0] = torch.tensor([0.0, 0.0, 4.2])
    scales[0] = torch.tensor([1.6, 1.6])
    rotations[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    opacities[0] = 0.95
    others[0, 0] = 0.9

    return dict(
        means3D=means3D.to(DEV),
        scales=scales.to(DEV),
        rotations=rotations.to(DEV),
        opacities=opacities.to(DEV),
        colors=colors.to(DEV),
        others=others.to(DEV),
    )


def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def get_triangles(means3D, scales, rotations):
    """The README's get_triangles(), with splat2world built inline from the
    scale/rotation pair instead of read off a GaussianModel."""
    P = means3D.shape[0]
    R = quat_to_rotmat(rotations)                       # (P, 3, 3)
    L = R[:, :, :2] * scales[:, None, :]                # (P, 3, 2)

    T = torch.zeros(P, 4, 4, device=means3D.device, dtype=means3D.dtype)
    T[:, :3, :2] = L
    T[:, :3, 3] = means3D
    T[:, 3, 3] = 1.0

    V = 4
    sigma3 = torch.as_tensor(
        [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, -1.0]],
        device=means3D.device, dtype=means3D.dtype) * 3
    sigma3 = torch.cat([sigma3, torch.ones_like(sigma3)], dim=-1)  # (V, 4)
    sigma3 = sigma3[None].repeat(P, 1, 1)                          # (P, V, 4)
    Te = T[:, None].expand(-1, V, -1, -1)                          # (P, V, 4, 4)

    v = (Te.reshape(-1, 4, 4) @ sigma3.reshape(-1, 4, 1))[..., :3, 0]
    indices = torch.arange(0, v.shape[0], device=means3D.device).reshape(P, V)
    f = torch.stack([indices[:, :3], indices[:, 1:]], dim=1).reshape(-1, 3).int()
    return v.contiguous(), f.contiguous()


def make_rays(H, W, fov=0.8):
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=DEV),
        torch.linspace(-1, 1, W, device=DEV),
        indexing="ij")
    d = torch.stack([xs * fov, ys * fov, torch.ones_like(xs)], dim=-1)
    d = d / d.norm(dim=-1, keepdim=True)
    o = torch.zeros(H, W, 3, device=DEV)
    return o.contiguous(), d.contiguous()


def make_settings(H, W, max_trace_depth=0, specular_threshold=0.0):
    eye = torch.eye(4, device=DEV)
    return SurfelTracingSettings(
        image_height=H, image_width=W, tanfovx=1.0, tanfovy=1.0,
        bg=torch.zeros(3, device=DEV),
        scale_modifier=1.0,
        viewmatrix=eye, projmatrix=eye,
        sh_degree=0,
        campos=torch.zeros(3, device=DEV),
        prefiltered=False, debug=False,
        max_trace_depth=max_trace_depth,
        specular_threshold=specular_threshold,
    )


def trace(tracer, scene, settings, H, W, requires_grad=False):
    means3D = scene["means3D"].clone().requires_grad_(requires_grad)
    scales = scene["scales"].clone().requires_grad_(requires_grad)
    rotations = scene["rotations"].clone().requires_grad_(requires_grad)
    opacities = scene["opacities"].clone().requires_grad_(requires_grad)
    colors = scene["colors"].clone().requires_grad_(requires_grad)
    others = scene["others"].clone().requires_grad_(requires_grad)
    grads3D = torch.zeros_like(means3D).requires_grad_(requires_grad)

    v, f = get_triangles(means3D.detach(), scales.detach(), rotations.detach())
    tracer.build_acceleration_structure(v, f, 1)

    ray_o, ray_d = make_rays(H, W)
    out = tracer(
        ray_o=ray_o, ray_d=ray_d, vertices=v,
        means3D=means3D, grads3D=grads3D,
        colors_precomp=colors, others_precomp=others,
        opacities=opacities, scales=scales, rotations=rotations,
        tracer_settings=settings)
    inputs = dict(means3D=means3D, grads3D=grads3D, scales=scales,
                  rotations=rotations, opacities=opacities, colors=colors,
                  others=others)
    return out, inputs


def scalar_loss(out):
    rgb, dpt, acc, norm = out[0], out[1], out[2], out[3]
    return (rgb.square().sum() + dpt.square().sum()
            + acc.square().sum() + norm.square().sum())


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_import():
    for sym in ("OptiXStateWrapper", "build_acceleration_structure",
                "trace_surfels", "trace_surfels_backward"):
        check(f"_C exports {sym}", hasattr(diff_surfel_tracing._C, sym))
    check("torch is a ROCm build", bool(torch.version.hip),
          f"torch.version.hip={torch.version.hip}")
    print(f"       device: {torch.cuda.get_device_name(0)}")


def check_forward(tracer, scene, H, W):
    settings = make_settings(H, W)
    out, _ = trace(tracer, scene, settings, H, W)
    rgb, dpt, acc, norm = out[0], out[1], out[2], out[3]

    check("forward rgb finite", bool(torch.isfinite(rgb).all()))
    check("forward depth finite", bool(torch.isfinite(dpt).all()))
    check("forward normal finite", bool(torch.isfinite(norm).all()))

    # The mirror surfel spans the whole frame, so every pixel accumulates
    # something and "acc > 0" says nothing. What distinguishes a real image
    # from a degenerate one here is the spread of coverage: pixels that a
    # near-opaque surfel actually covers versus pixels that only catch the
    # tail of a Gaussian.
    covered = acc > 0.5
    hit = covered.float().mean().item()
    check("forward has genuine hits and misses", 0.05 < hit < 0.95,
          f"covered fraction {hit:.3f}, acc range "
          f"[{acc.min():.3f}, {acc.max():.3f}]")

    # dpt is the coverage-weighted sum of hit distances, not a distance, so it
    # only becomes a depth after dividing by acc -- and only where acc is large
    # enough for that quotient to mean anything.
    hit_dpt = (dpt / acc.clamp_min(1e-6))[covered]
    check("forward hit depths plausible",
          bool((hit_dpt > 0.5).all() and (hit_dpt < 10.0).all()),
          f"depth range [{hit_dpt.min():.3f}, {hit_dpt.max():.3f}]")

    out2, _ = trace(tracer, scene, settings, H, W)
    check("forward is bit-identical on rerun",
          bool(torch.equal(rgb, out2[0]) and torch.equal(dpt, out2[1])))
    return rgb


# A surfel can legitimately receive no position gradient without any geometry
# being lost: a disk seen edge-on covers no pixel. Measured on this scene, 62 of
# 64 are traced and the two that are not (5 and 25) are the two most edge-on,
# with |cos(normal, view)| of 0.003 and 0.015 against a median of 0.41. The
# margin covers those and is still far tighter than any missing-geometry defect:
# a BVH that drops the upper half of every 64-wide wavefront leaves 31 of 64.
GEOMETRY_OCCLUSION_MARGIN = 4


def check_geometry_complete(scene, touched):
    """A BVH built from only part of the scene still renders a plausible image
    with finite gradients -- nothing else in this harness would notice. This is
    the check that does: every surfel centre projects inside the frame, so a
    complete acceleration structure has to trace all but the edge-on few."""
    means3D = scene["means3D"]
    P = means3D.shape[0]
    x, y, z = means3D[:, 0], means3D[:, 1], means3D[:, 2]
    # make_rays() spans x and y over [-fov, fov] at z = 1.
    in_frame = (z > 0) & (x.abs() <= 0.8 * z) & (y.abs() <= 0.8 * z)
    check("every surfel centre is inside the frame", bool(in_frame.all()),
          f"{int(in_frame.sum())}/{P}")

    missing = (~touched & in_frame).nonzero().flatten().tolist()
    check("the traced geometry is complete",
          len(missing) <= GEOMETRY_OCCLUSION_MARGIN,
          f"{int(touched.sum())}/{P} surfels traced"
          + (f", missing {missing}" if missing else ""))


def check_backward(tracer, scene, H, W):
    settings = make_settings(H, W)
    out, inputs = trace(tracer, scene, settings, H, W, requires_grad=True)
    scalar_loss(out).backward()

    for name, t in inputs.items():
        check(f"backward grad_{name} finite", bool(torch.isfinite(t.grad).all()))

    for name in ("means3D", "scales", "rotations", "opacities", "colors"):
        g = inputs[name].grad
        check(f"backward grad_{name} nonzero", bool(g.abs().max() > 0),
              f"max |g| {g.abs().max():.3e}")

    # The densification gradient is a depth-scaled copy of the position
    # gradient, accumulated in the kernel. A back end that never writes it
    # returns finite zeros and passes a naive finiteness check.
    gg = inputs["grads3D"].grad
    gm = inputs["means3D"].grad
    check("backward grad_grads3D nonzero", bool(gg.abs().max() > 0),
          f"max |g| {gg.abs().max():.3e}")
    touched = gm.abs().sum(-1) > 0
    check_geometry_complete(scene, touched)
    check("grad_grads3D is nonzero wherever grad_means3D is",
          bool((gg[touched].abs().sum(-1) > 0).all()),
          f"{int(touched.sum())} surfels received a position gradient")
    # Per surfel, grads3D is the same per-hit position gradient reweighted by
    # the hit distance, so the two vectors point the same way for a surfel
    # whose hits are at a similar depth and diverge where near and far hits
    # cancel differently. The median over surfels is the honest statistic; one
    # cosine over the flattened stack is dominated by whichever surfel has the
    # largest gradient.
    per = torch.nn.functional.cosine_similarity(gg[touched], gm[touched], dim=-1)
    med = per.median().item()
    check("grad_grads3D tracks grad_means3D", med > 0.9,
          f"median per-surfel cosine {med:.4f} over {int(touched.sum())} surfels")


# The loss is O(1e4) in float32, so a central difference divided by 2*eps
# amplifies the ~1e-3 rounding of each term by 1/(2*eps). At eps=1e-4 that is
# larger than the derivative itself and the quotient is pure noise; at 1e-3 the
# rounding is well below the signal and the truncation error is still small.
FD_EPS = 1e-3
# A ray tracer's loss is piecewise smooth: moving geometry moves silhouettes,
# and a single directional derivative can land on one of those steps. Several
# directions turn the comparison into a correlation, where an isolated step
# perturbs the fit instead of deciding it.
FD_DIRECTIONS = 6


def fd_gradcheck(tracer, scene, H, W, name, lo, hi, finite_only=False):
    settings = make_settings(H, W)
    out, inputs = trace(tracer, scene, settings, H, W, requires_grad=True)
    scalar_loss(out).backward()
    analytic = inputs[name].grad.clone()

    base = scene[name].clone()
    fds, ans = [], []
    for k in range(FD_DIRECTIONS):
        torch.manual_seed(1234 + k)
        direction = torch.randn_like(analytic)
        direction /= direction.norm()
        losses = []
        for sign in (+1, -1):
            scene[name] = (base + sign * FD_EPS * direction).contiguous()
            out, _ = trace(tracer, scene, settings, H, W)
            losses.append(scalar_loss(out).item())
        fds.append((losses[0] - losses[1]) / (2 * FD_EPS))
        ans.append((analytic * direction).sum().item())
    scene[name] = base

    fd = torch.tensor(fds, dtype=torch.float64)
    an = torch.tensor(ans, dtype=torch.float64)
    if finite_only:
        check(f"finite difference {name} finite",
              bool(torch.isfinite(fd).all() and torch.isfinite(an).all()),
              f"|fd| max {fd.abs().max():.4e} |analytic| max {an.abs().max():.4e}")
        return
    cos = torch.nn.functional.cosine_similarity(fd, an, dim=0).item()
    slope = (fd @ an / (fd @ fd)).item()
    check(f"finite difference {name} matches analytic",
          cos > 0.9 and lo <= slope <= hi,
          f"cosine {cos:.4f} slope {slope:.4f} over {FD_DIRECTIONS} directions")


def check_reflection(tracer, scene, H, W):
    settings = make_settings(H, W, max_trace_depth=1, specular_threshold=0.1)
    out, inputs = trace(tracer, scene, settings, H, W, requires_grad=True)
    rgb = out[0]
    check("reflected bounce rgb finite", bool(torch.isfinite(rgb).all()))

    flat = make_settings(H, W, max_trace_depth=0, specular_threshold=0.0)
    out0, _ = trace(tracer, scene, flat, H, W)
    delta = (rgb - out0[0]).abs().max().item()
    check("reflected bounce changes the image", delta > 1e-5,
          f"max |delta rgb| {delta:.3e}")

    scalar_loss(out).backward()
    for name, t in inputs.items():
        check(f"reflected bounce grad_{name} finite",
              bool(torch.isfinite(t.grad).all()))
    return rgb


# The acceleration-structure build sorts one key per triangle pair, and the
# tessellation gives one pair per surfel, so the surfel count is the sorted
# element count. Below 3072 keys the radix sort runs a single-pass kernel in one
# work group; above it a multi-block kernel with a different reordering scheme
# takes over, and that one has no other route into this harness -- every check
# above runs at 64 surfels, two orders of magnitude below the threshold. 4096
# crosses it with one block, 16384 adds the cross-block lookback.
SCALE_SURFEL_COUNTS = (4096, 16384)


def check_scale(H, W):
    """Trace scenes large enough to reach the multi-block sort in the build.

    Deliberately NOT the geometry-completeness criterion: a slab of this many
    disks occludes itself, so most surfels are legitimately never hit, and
    counting them says nothing. What has meaning at this size is that the build
    and both passes finish -- a wrong sort here corrupts the indices of a global
    write and takes the process down with a memory access fault, which no
    assertion can catch -- and that the result is finite and non-degenerate.
    """
    for P in SCALE_SURFEL_COUNTS:
        tracer = SurfelTracer().to(DEV)
        tracer.train()
        scene = make_scene(P=P, seed=1)
        settings = make_settings(H, W)
        out, inputs = trace(tracer, scene, settings, H, W, requires_grad=True)
        scalar_loss(out).backward()

        rgb, acc = out[0], out[2]
        finite = bool(torch.isfinite(rgb).all()) and all(
            bool(torch.isfinite(t.grad).all()) for t in inputs.values())
        check(f"{P} surfels: forward and backward are finite", finite,
              f"rgb mean {rgb.mean().item():.5f}")

        covered = (acc > 0.5).float().mean().item()
        check(f"{P} surfels: the image has genuine hits and misses",
              0.05 < covered < 0.999, f"covered fraction {covered:.3f}")

        touched = int((inputs["means3D"].grad.abs().sum(-1) > 0).sum())
        check(f"{P} surfels: the scene is genuinely traced", touched > 0,
              f"{touched}/{P} surfels received a position gradient")


def check_cold_cache(tracer_factory, scene, H, W, warm_rgb):
    cache = os.path.join(os.path.dirname(diff_surfel_tracing.__file__),
                         "hiprt_cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
    tracer = tracer_factory()
    settings = make_settings(H, W)
    out, _ = trace(tracer, scene, settings, H, W)
    check("cold compilation cache reproduces the warm image bit for bit",
          bool(torch.equal(out[0], warm_rgb)))
    check("compilation cache repopulated", os.path.isdir(cache)
          and len(os.listdir(cache)) > 0)


def main():
    if not torch.cuda.is_available():
        print("no GPU visible")
        return 2

    H = W = 96
    scene = make_scene()
    tracer = SurfelTracer().to(DEV)
    tracer.train()

    print("== import and API ==")
    check_import()

    print("\n== forward ==")
    warm_rgb = check_forward(tracer, scene, H, W)

    print("\n== backward ==")
    check_backward(tracer, scene, H, W)

    print("\n== finite differences ==")
    # Colors enter the composite linearly, so that one is the exact gate.
    fd_gradcheck(tracer, scene, H, W, "colors", 0.95, 1.05)
    fd_gradcheck(tracer, scene, H, W, "opacities", 0.5, 1.8)
    fd_gradcheck(tracer, scene, H, W, "means3D", 0.5, 1.8)
    fd_gradcheck(tracer, scene, H, W, "scales", 0.5, 1.8)
    # Rotations are only checked for finiteness: the quaternion is renormalized
    # inside the kernel, so the component along the quaternion itself is a null
    # direction and a difference quotient along a random direction measures
    # that null space as much as the gradient.
    fd_gradcheck(tracer, scene, H, W, "rotations", 0, 0, finite_only=True)

    print("\n== reflected bounce ==")
    check_reflection(tracer, scene, H, W)

    print("\n== scale ==")
    check_scale(H, W)

    print("\n== cold compilation cache ==")
    check_cold_cache(lambda: SurfelTracer().to(DEV), scene, H, W, warm_rgb)

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("failed: " + ", ".join(FAILURES))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
