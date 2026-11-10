import os
import shutil
import site
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
USE_HIP = bool(torch.version.hip)


if not USE_HIP:
    # Determine the OptiX SDK path
    # If there is a optix directory in the `third_party` directory, use it as the OptiX SDK path
    # Otherwise, use the environment variable `OPTIX_HOME`
    # If both are not set, raise an error
    OPTIX_HOME = os.path.join(ROOT, "third_party/optix")
    if not os.path.exists(OPTIX_HOME):
        OPTIX_HOME = os.environ.get('OPTIX_HOME')
        if OPTIX_HOME is None or not os.path.exists(OPTIX_HOME):
            raise ValueError("Please set the OPTIX_HOME environment variable to the path to the OptiX SDK")
    OPTIX_HOME = os.path.join(OPTIX_HOME, "include")
    print(f"Using OptiX SDK at {OPTIX_HOME}")
else:
    # Determine the HIP RT SDK path, mirroring the OptiX resolution above: the
    # `third_party/hiprt` submodule if it is checked out, otherwise HIPRT_HOME.
    HIPRT_HOME = os.path.join(ROOT, "third_party/hiprt")
    if not os.path.exists(os.path.join(HIPRT_HOME, "hiprt")):
        HIPRT_HOME = os.environ.get('HIPRT_HOME')
        if HIPRT_HOME is None or not os.path.exists(HIPRT_HOME):
            raise ValueError("Please set the HIPRT_HOME environment variable to the path to the HIP RT SDK")
    print(f"Using HIP RT SDK at {HIPRT_HOME}")

    OROCHI_ROOT = os.path.join(HIPRT_HOME, "contrib", "Orochi")
    HIPRT_LIB_DIR = os.path.join(HIPRT_HOME, "dist", "bin", "Release")
    HIPRT_LIB = "hiprt0300164"
    GLUE_BUILD_DIR = os.path.join(ROOT, "build", "hiprt_glue")

    # On Windows, build the glue with clang++/llvm-ar from the ROCm SDK devel
    # tree rather than g++/ar, and emit a Windows-native import library.
    _IS_WIN_HIP = os.name == 'nt'
    if _IS_WIN_HIP:
        _rocm_devel = None
        for _sp in list(torch.__path__) + list(site.getsitepackages()):
            _candidate = os.path.join(os.path.dirname(_sp), "_rocm_sdk_devel")
            if os.path.isdir(_candidate):
                _rocm_devel = _candidate
                break
        if _rocm_devel is None:
            raise RuntimeError("Cannot find _rocm_sdk_devel alongside torch")
        ROCM_INCLUDE = os.path.join(_rocm_devel, "include")
        CLANGPP = os.path.join(_rocm_devel, "lib", "llvm", "bin", "clang++.exe")
        LLVM_AR = os.path.join(_rocm_devel, "lib", "llvm", "bin", "llvm-ar.exe")
        GLUE_LIB = os.path.join(GLUE_BUILD_DIR, "hiprt_tracer_glue.lib")
    else:
        ROCM_INCLUDE = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include")
        CLANGPP = "g++"
        LLVM_AR = "ar"
        GLUE_LIB = os.path.join(GLUE_BUILD_DIR, "libhiprt_tracer_glue.a")

    # The HIP RT / Orochi glue is compiled as a STANDALONE static library rather
    # than as part of the extension, because Orochi's hipew driver loader
    # redeclares the HIP driver API and conflicts with torch's
    # <hip/hip_runtime.h>; the two cannot share a translation unit. The torch
    # extension reaches the glue through the plain C++ / POD / void* interface in
    # hiprt_tracer/hiprt_wrapper.h.
    GLUE_SOURCES = [
        os.path.join(ROOT, "hiprt_tracer", "hiprt_wrapper.cpp"),
        os.path.join(OROCHI_ROOT, "Orochi", "Orochi.cpp"),
        os.path.join(OROCHI_ROOT, "Orochi", "OrochiUtils.cpp"),
        os.path.join(OROCHI_ROOT, "contrib", "hipew", "src", "hipew.cpp"),
        os.path.join(OROCHI_ROOT, "contrib", "cuew", "src", "cuew.cpp"),
    ]

    # Device kernel source plus the headers it includes, shipped flat into the
    # package so HIP RT can compile them at runtime. This is the ROCm analogue of
    # the OptiX PTX that the CMake step below produces on NVIDIA.
    PKG_RUNTIME_HEADERS = [
        "hiprt_tracer/kernels.h",
        "optix_tracer/params.h",
        "optix_tracer/config.h",
        "optix_tracer/auxiliary.h",
    ]

    # The parts of a HIP RT checkout its runtime compiler opens, relative to
    # HIPRT_PATH. Everything else in that checkout is host-side build material.
    HIPRT_RUNTIME_SUBDIRS = [
        "hiprt",
        "contrib/Orochi/ParallelPrimitives",
    ]


def _build_glue_static_lib():
    os.makedirs(GLUE_BUILD_DIR, exist_ok=True)
    common = [
        CLANGPP, "-c", "-std=c++17", "-O2",
        "-D__USE_HIP__", "-DHIPRT_PUBLIC_REPO",
        "-I" + ROOT, "-I" + HIPRT_HOME, "-I" + OROCHI_ROOT, "-I" + ROCM_INCLUDE,
    ]
    if _IS_WIN_HIP:
        # Match torch's objects: dynamic CRT (clang++ defaults to /MT, which
        # trips LNK2038 RuntimeLibrary mismatch against the torch extension).
        common += ["-fms-runtime-lib=dll", "-D_CRT_SECURE_NO_WARNINGS"]
        ext = ".obj"
    else:
        common += ["-fPIC"]
        ext = ".o"
    objs = []
    for src in GLUE_SOURCES:
        obj = os.path.join(GLUE_BUILD_DIR, os.path.basename(src) + ext)
        subprocess.check_call(common + [src, "-o", obj])
        objs.append(obj)
    if os.path.exists(GLUE_LIB):
        os.remove(GLUE_LIB)
    subprocess.check_call([LLVM_AR, "rcs", GLUE_LIB] + objs)


def _patch_hipify_ignore_hiprt():
    """Keep the HIP RT and Orochi trees out of torch's build-time hipify: they
    are already HIP sources and must not be rewritten."""
    from torch.utils.hipify import hipify_python

    patterns = [os.path.join(HIPRT_HOME, "*"), HIPRT_HOME + "*"]
    orig_hipify = hipify_python.hipify

    def hipify_no_hiprt(*args, **kwargs):
        kwargs["ignores"] = list(kwargs.get("ignores", ())) + patterns
        kwargs["header_include_dirs"] = [
            d for d in kwargs.get("header_include_dirs", [])
            if os.path.abspath(d) != os.path.abspath(HIPRT_HOME)
        ]
        return orig_hipify(*args, **kwargs)

    hipify_python.hipify = hipify_no_hiprt


def _stage_runtime_files_into_pkg():
    """Place the runtime-compiled kernel source and the HIP RT root next to
    __init__.py in the source package directory. SurfelTracer resolves its
    pkg_dir from __file__, so this serves an editable install directly, and
    package_data carries the same files into a wheel."""
    pkg = os.path.join(ROOT, "diff_surfel_tracing")
    for h in PKG_RUNTIME_HEADERS:
        shutil.copy(os.path.join(ROOT, h), pkg)
    # HIP RT reads its own BVH-builder kernel sources from HIPRT_PATH at runtime,
    # so they have to travel with the package. Only these two subtrees are read:
    # hiprt/ for the builder and traversal kernels, and Orochi's ParallelPrimitives
    # for the radix sort the builders use (hiprt/impl/RadixSort.cpp). Copying the
    # whole checkout instead would drag in ~190 MB of history, build output and
    # vendored test dependencies that are never opened at runtime.
    dst_hiprt = os.path.join(pkg, "hiprt_root")
    if os.path.isdir(dst_hiprt):
        shutil.rmtree(dst_hiprt)
    for sub in HIPRT_RUNTIME_SUBDIRS:
        src = os.path.join(HIPRT_HOME, *sub.split("/"))
        if not os.path.isdir(src):
            raise RuntimeError(f"HIPRT_HOME is missing {sub}: {HIPRT_HOME}")
        shutil.copytree(src, os.path.join(dst_hiprt, *sub.split("/")))
    # The HIP RT runtime library travels with the package too, so an installed
    # package does not depend on the build tree; _C finds it next to itself.
    if _IS_WIN_HIP:
        hiprt_dll = os.path.join(HIPRT_LIB_DIR, HIPRT_LIB + ".dll")
        if os.path.exists(hiprt_dll):
            shutil.copy(hiprt_dll, pkg)
    else:
        shutil.copy(os.path.join(HIPRT_LIB_DIR, "lib" + HIPRT_LIB + ".so"), pkg)


class CustomBuildPy(build_py):
    # build_py collects package_data before build_ext runs, so the runtime
    # files have to be staged here for a wheel or a regular install to carry them.
    def run(self):
        _stage_runtime_files_into_pkg()
        self.__dict__.pop("data_files", None)
        super().run()


class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        if USE_HIP:
            _build_glue_static_lib()
            _stage_runtime_files_into_pkg()
            super().build_extensions()
            return

        # Record the python package source directory
        pkg_source = ROOT

        # Run the original build_extensions
        super().build_extensions()

        # Use CMake to build the OptiX tracing kernel ptx files
        os.system(f'mkdir -p {pkg_source}/build && cd {pkg_source}/build && cmake .. && make')
        pkg_target = site.getsitepackages()[0] + '/diff_surfel_tracing'

        # Create the target directory if it does not exist
        if not os.path.exists(pkg_target):
            os.makedirs(pkg_target, exist_ok=True)

        # Copy the `.ptx` files to the python package
        os.system(f'cp {pkg_source}/build/ptx/*.ptx {pkg_target}')


if USE_HIP:
    _patch_hipify_ignore_hiprt()

    # On Windows, MSVC cl.exe compiles the .cpp sources but cannot link
    # c10::ValueError(SourceLocation, string) out of the Clang-built c10.dll
    # (it is absent from the MSVC import lib). Compiling the torch-facing TUs
    # as .cu routes them through hipcc/amdclang++, which shares c10.dll's ABI.
    _EXT_SOURCES = ["trace_surfels.cpp", "ext.cpp"]
    if _IS_WIN_HIP:
        for _src in list(_EXT_SOURCES):
            _dst = _src.replace(".cpp", "_winhip.cu")
            shutil.copy(os.path.join(ROOT, _src), os.path.join(ROOT, _dst))
        _EXT_SOURCES = [s.replace(".cpp", "_winhip.cu") for s in _EXT_SOURCES]

    _ext_modules = [
        CUDAExtension(
            name="diff_surfel_tracing._C",
            sources=_EXT_SOURCES,
            include_dirs=[ROOT],
            extra_objects=[GLUE_LIB],
            library_dirs=[HIPRT_LIB_DIR],
            # hipew.cpp calls GetFileVersionInfoA/VerQueryValueA from Version.lib.
            libraries=[HIPRT_LIB] + (["version"] if _IS_WIN_HIP else []),
            extra_link_args=[] if _IS_WIN_HIP else ["-Wl,-rpath,$ORIGIN"],
        ),
    ]
    _package_data = {"diff_surfel_tracing": [
        "kernels.h", "params.h", "config.h", "auxiliary.h", "hiprt_root/**/*", "*.dll",
        "lib" + HIPRT_LIB + ".so",
    ]}
    _cmdclass = {'build_py': CustomBuildPy}
else:
    _ext_modules = [
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
    ]
    _package_data = {}
    _cmdclass = {}


# Setup for the python package
setup(
    name="diff_surfel_tracing",
    packages=['diff_surfel_tracing'],
    version='0.0.1',
    package_data=_package_data,
    ext_modules=_ext_modules,
    cmdclass={
        **_cmdclass,
        'build_ext': CustomBuildExtension
    }
)
