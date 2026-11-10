#include <torch/extension.h>

#include "trace_surfels.h"
#ifdef USE_ROCM
// ROCm/HIP build: the OptiX state wrapper is reimplemented on HIP RT. The class
// name OptiXStateWrapper is kept so ext.cpp and the Python autograd wrapper
// reference the same symbol on both back ends.
#include "hiprt_tracer/hiprt_wrapper.h"
#else
#include "optix_tracer/optix_wrapper.h"
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<OptiXStateWrapper>(m, "OptiXStateWrapper").def(pybind11::init<const std::string &>());
  m.def("build_acceleration_structure", &BuildAccelerationStructure);
  m.def("trace_surfels", &TraceSurfelsCUDA);
  m.def("trace_surfels_backward", &TraceSurfelsBackwardCUDA);
}
