import os
import torch
import torch.nn as nn
from typing import NamedTuple

# On Windows the HIP RT runtime library is staged next to this __init__.py by
# setup.py, so register the package directory as a DLL search path before _C is
# imported. Orochi's loader uses plain LoadLibraryA, which follows the legacy
# search order and ignores AddDllDirectory, so the ROCm SDK bin directories are
# prepended to PATH as well.
if os.name == 'nt' and torch.version.hip:
    import sys as _sys
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    os.add_dll_directory(_pkg_dir)
    _venv_sp = os.path.normpath(os.path.join(os.path.dirname(_sys.executable), '..', 'Lib', 'site-packages'))
    _prepend = []
    for _sdk in ('_rocm_sdk_core', '_rocm_sdk_devel'):
        _d = os.path.join(_venv_sp, _sdk, 'bin')
        if os.path.isdir(_d):
            _prepend.append(_d)
            os.add_dll_directory(_d)
    if _prepend:
        os.environ['PATH'] = os.pathsep.join(_prepend) + os.pathsep + os.environ.get('PATH', '')
    _rocm_devel = os.path.join(_venv_sp, '_rocm_sdk_devel')
    if os.path.isdir(_rocm_devel) and not os.environ.get('ROCM_PATH'):
        os.environ['ROCM_PATH'] = _rocm_devel
    del _sys, _pkg_dir, _venv_sp, _prepend, _rocm_devel, _sdk, _d

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def trace_surfels(optix_context,
                  training,
                  start_from_first,
                  ray_o,
                  ray_d,
                  vertices,
                  means3D,
                  grads3D,
                  shs,
                  colors_precomp,
                  others_precomp,
                  opacities,
                  scales,
                  rotations,
                  cov3Ds_precomp,
                  tracer_settings
                  ):

    # Invoke the autograd function
    return _TraceSurfels.apply(optix_context,
                               training,
                               start_from_first,
                               ray_o,
                               ray_d,
                               vertices,
                               means3D,
                               grads3D,
                               shs,
                               colors_precomp,
                               others_precomp,
                               opacities,
                               scales,
                               rotations,
                               cov3Ds_precomp,
                               tracer_settings)


class _TraceSurfels(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                optix_context,
                training,
                start_from_first,
                ray_o,
                ray_d,
                vertices,
                means3D,
                grads3D,
                shs,
                colors_precomp,
                others_precomp,
                opacities,
                scales,
                rotations,
                cov3Ds_precomp,
                tracer_settings,
                ):

        # Restructure arguments the way that the C++ lib expects them
        args = (optix_context,
                training,
                start_from_first,
                ray_o,
                ray_d,
                vertices,
                tracer_settings.bg,
                means3D,
                shs,
                tracer_settings.sh_degree,
                colors_precomp,
                others_precomp,
                opacities,
                scales,
                tracer_settings.scale_modifier,
                rotations,
                cov3Ds_precomp,
                tracer_settings.viewmatrix,
                tracer_settings.projmatrix,
                tracer_settings.campos,
                tracer_settings.prefiltered,
                tracer_settings.debug,
                tracer_settings.max_trace_depth,
                tracer_settings.specular_threshold)

        # Invoke C++/CUDA/OptiX tracer
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                out_rgb, out_dpt, out_acc, out_norm, out_dist, out_aux, mid_val, a_weights = _C.trace_surfels(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            out_rgb, out_dpt, out_acc, out_norm, out_dist, out_aux, mid_val, a_weights = _C.trace_surfels(*args)

        # Keep relevant tensors for backward
        ctx.tracer_settings = tracer_settings
        ctx.optix_context = optix_context
        ctx.start_from_first = start_from_first
        ctx.save_for_backward(ray_o, ray_d, vertices, means3D, shs, colors_precomp, others_precomp, opacities, scales, rotations, cov3Ds_precomp,
                              out_rgb, out_dpt, out_acc, out_norm, out_dist, out_aux, mid_val)

        # Return the per-Gaussian hit counter for training gradient filtering
        acc_wet = a_weights if a_weights is not None else torch.zeros_like(means3D[:, 0], dtype=means3D.dtype, device=means3D.device)

        return out_rgb, out_dpt, out_acc, out_norm, out_dist[..., :1].clone(), out_aux, mid_val, acc_wet

    @staticmethod
    def backward(ctx, grad_out_rgb, grad_out_dpt, grad_out_acc, grad_out_norm, grad_out_dist, grad_out_aux, grad_mid_val, grad_acc_wet):

        # Restore necessary values from context
        tracer_settings = ctx.tracer_settings
        optix_context = ctx.optix_context
        start_from_first = ctx.start_from_first
        ray_o, ray_d, vertices, means3D, shs, colors_precomp, others_precomp, opacities, scales, rotations, cov3Ds_precomp, \
            out_rgb, out_dpt, out_acc, out_norm, out_dist, out_aux, mid_val = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (optix_context,
                start_from_first,
                ray_o,
                ray_d,
                vertices,
                tracer_settings.bg,
                means3D,
                shs,
                tracer_settings.sh_degree,
                colors_precomp,
                others_precomp,
                opacities,
                scales,
                tracer_settings.scale_modifier,
                rotations,
                cov3Ds_precomp,
                tracer_settings.viewmatrix,
                tracer_settings.projmatrix,
                tracer_settings.campos,
                tracer_settings.prefiltered,
                tracer_settings.debug,
                tracer_settings.max_trace_depth,
                tracer_settings.specular_threshold,
                out_rgb,
                out_dpt,
                out_acc,
                out_norm,
                out_dist,
                out_aux,
                mid_val,
                grad_out_rgb,
                grad_out_dpt,
                grad_out_acc,
                grad_out_norm,
                grad_out_dist,
                grad_out_aux)

        # Compute gradients for relevant tensors by invoking backward method
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_ray_o, grad_ray_d, grad_means3D, grad_grads3D, grad_shs, grad_colors_precomp, grad_others_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp = _C.trace_surfels_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_ray_o, grad_ray_d, grad_means3D, grad_grads3D, grad_shs, grad_colors_precomp, grad_others_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp = _C.trace_surfels_backward(*args)

        grads = (
            None,
            None,
            None,
            grad_ray_o,
            grad_ray_d,
            None,
            grad_means3D,
            grad_grads3D,
            grad_shs,
            grad_colors_precomp,
            grad_others_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class SurfelTracingSettings(NamedTuple):
    image_height: int  # no use, only for compatibility
    image_width: int  # no use, only for compatibility
    tanfovx: float  # no use, only for compatibility
    tanfovy: float  # no use, only for compatibility
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    max_trace_depth: int = 0
    specular_threshold: float = 0.0


class SurfelTracer(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

        # Locate the package directory, which holds the OptiX PTX modules on
        # NVIDIA and the runtime-compiled kernel source on AMD. Resolve it from
        # this module's own location so it is correct for an editable install
        # too, not only for a wheel installed into site-packages.
        self.pkg_dir = os.path.dirname(os.path.abspath(__file__))

        # HIP RT compiles its BVH builder and traversal kernels at runtime and
        # reads their sources from HIPRT_PATH. Point it at the HIP RT root
        # staged inside the package so the tracer is self-contained. No effect
        # on the NVIDIA build.
        hiprt_root = os.path.join(self.pkg_dir, 'hiprt_root')
        if os.path.isdir(hiprt_root) and not os.environ.get('HIPRT_PATH'):
            os.environ['HIPRT_PATH'] = hiprt_root

        # Create the tracer context. The C++ class name is the same on both
        # back ends so the binding is unchanged.
        self.optix_context = _C.OptiXStateWrapper(self.pkg_dir)

    def build_acceleration_structure(self,
                                     vertices: torch.Tensor,
                                     triangles: torch.Tensor,
                                     rebuild: bool = 1,
                                     ):
        # Invoke C++/CUDA/OptiX acceleration structure building routine
        return _C.build_acceleration_structure(self.optix_context, vertices, triangles, rebuild)

    def forward(self,
                ray_o: torch.Tensor,
                ray_d: torch.Tensor,
                vertices: torch.Tensor,
                means3D: torch.Tensor,
                grads3D: torch.Tensor,
                shs: torch.Tensor = None,
                colors_precomp: torch.Tensor = None,
                others_precomp: torch.Tensor = None,
                opacities: torch.Tensor = None,
                scales: torch.Tensor = None,
                rotations: torch.Tensor = None,
                cov3D_precomp: torch.Tensor = None,
                tracer_settings: SurfelTracingSettings = None,
                start_from_first: bool = True
                ):

        # Check if colors or SHs are provided
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        # Check if scales/rotations or cov3D_precomp is provided
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # Create dummy color tracing input
        if shs is None: shs = torch.Tensor([]).cuda()
        if colors_precomp is None: colors_precomp = torch.Tensor([]).cuda()
        if others_precomp is None: others_precomp = torch.Tensor([]).cuda()
        # Create dummy tracing transformation matrix input
        if scales is None: scales = torch.Tensor([]).cuda()
        if rotations is None: rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None: cov3D_precomp = torch.Tensor([]).cuda()
        # Create dummy vertices input
        if vertices is None: vertices = torch.Tensor([]).cuda()

        # Invoke C++/CUDA/OptiX tracing routine
        return trace_surfels(self.optix_context,
                             self.training,
                             start_from_first,
                             ray_o,
                             ray_d,
                             vertices,
                             means3D,
                             grads3D,
                             shs,
                             colors_precomp,
                             others_precomp,
                             opacities,
                             scales,
                             rotations,
                             cov3D_precomp,
                             tracer_settings)
