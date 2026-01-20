import os
import site
import shutil
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME


# Determine the OptiX SDK path
# If there is a optix directory in the `third_party` directory, use it as the OptiX SDK path
# Otherwise, use the environment variable `OPTIX_HOME`
# If both are not set, raise an error
OPTIX_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/optix")
if not os.path.exists(OPTIX_HOME):
    OPTIX_HOME = os.environ.get('OPTIX_HOME')
    if OPTIX_HOME is None or not os.path.exists(OPTIX_HOME):
        raise ValueError("Please set the OPTIX_HOME environment variable to the path to the OptiX SDK")
OPTIX_HOME = os.path.join(OPTIX_HOME, "include")
print(f"Using OptiX SDK at {OPTIX_HOME}")

def _run_checked(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _get_nvcc():
    if CUDA_HOME:
        nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.exists(nvcc):
            return nvcc
    return "nvcc"


def _compile_ptx(src_path, dst_path, include_dirs):
    nvcc = _get_nvcc()
    cmd = [nvcc, "-ptx", "-O3", "-std=c++17"]
    try:
        from torch.utils.cpp_extension import _get_cuda_arch_flags
        cmd += _get_cuda_arch_flags()
    except Exception:
        pass
    for include_dir in include_dirs:
        cmd += ["-I", include_dir]
    cmd += ["-o", dst_path, src_path]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    _run_checked(cmd)


# Custom build extension to build the OptiX tracing kernel
class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        # Record the python package source directory
        pkg_source = os.path.dirname(os.path.abspath(__file__))

        # Run the original build_extensions
        super().build_extensions()

        # Use CMake to build the OptiX tracing kernel ptx files (fallback to nvcc if needed)
        build_dir = os.path.join(pkg_source, "build")
        os.makedirs(build_dir, exist_ok=True)
        cmake_ok = True
        try:
            _run_checked(["cmake", ".."], cwd=build_dir)
            _run_checked(["cmake", "--build", "."], cwd=build_dir)
        except Exception as exc:
            cmake_ok = False
            print(f"Warning: CMake PTX build failed, falling back to nvcc: {exc}")

        ptx_dir = os.path.join(build_dir, "ptx")
        forward_ptx = os.path.join(ptx_dir, "forward.ptx")
        backward_ptx = os.path.join(ptx_dir, "backward.ptx")
        if not cmake_ok or not (os.path.exists(forward_ptx) and os.path.exists(backward_ptx)):
            include_dirs = [
                OPTIX_HOME,
                os.path.join(pkg_source, "optix_tracer"),
            ]
            _compile_ptx(
                os.path.join(pkg_source, "optix_tracer", "forward.cu"),
                forward_ptx,
                include_dirs,
            )
            _compile_ptx(
                os.path.join(pkg_source, "optix_tracer", "backward.cu"),
                backward_ptx,
                include_dirs,
            )

        pkg_targets = []
        if getattr(self, "build_lib", None):
            pkg_targets.append(os.path.join(self.build_lib, "diff_surfel_tracing"))
        try:
            pkg_targets.append(os.path.join(site.getsitepackages()[0], "diff_surfel_tracing"))
        except Exception:
            pass

        for pkg_target in pkg_targets:
            os.makedirs(pkg_target, exist_ok=True)
            shutil.copy2(forward_ptx, pkg_target)
            shutil.copy2(backward_ptx, pkg_target)


# Setup for the python package
setup(
    name="diff_surfel_tracing",
    packages=['diff_surfel_tracing'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_tracing._C",
            sources=[
                "optix_tracer/common.cpp",
                "optix_tracer/optix_wrapper.cpp",
                "trace_surfels.cpp",
                "ext.cpp"
            ],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]},
            include_dirs=[OPTIX_HOME]
        ),
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)
