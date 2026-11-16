#include "hexplane_sample.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("sample_plane", &hexplane_sample_plane_cuda,
               pybind11::arg("coords"), pybind11::arg("plane"),
               pybind11::arg("axis_x"), pybind11::arg("axis_y"),
               pybind11::arg("probe_512") = false);
    module.def("sample_plane_out", &hexplane_sample_plane_out_cuda,
               pybind11::arg("coords"), pybind11::arg("plane"),
               pybind11::arg("output"), pybind11::arg("axis_x"),
               pybind11::arg("axis_y"), pybind11::arg("probe_512") = false);
    module.def("resources", &hexplane_sample_resources_cuda);
    module.def("sample_planes", &hexplane_sample_planes_cuda,
               pybind11::arg("coords"), pybind11::arg("planes"), pybind11::arg("axes"));
    module.def("sample_planes_out", &hexplane_sample_planes_out_cuda,
               pybind11::arg("coords"), pybind11::arg("planes"),
               pybind11::arg("axes"), pybind11::arg("outputs"));
    module.def("batch_resources", &hexplane_sample_batch_resources_cuda);
    module.def("reduce_features", &hexplane_reduce_features_cuda,
               pybind11::arg("samples"), pybind11::arg("threads") = 256);
    module.def("reduce_features_out", &hexplane_reduce_features_out_cuda,
               pybind11::arg("samples"), pybind11::arg("output"),
               pybind11::arg("threads") = 256);
    module.def("reduction_resources", &hexplane_reduce_resources_cuda);
}
