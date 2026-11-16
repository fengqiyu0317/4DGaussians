#include <torch/extension.h>

std::vector<torch::Tensor> gsplat_mixed_forward(
    const std::vector<torch::Tensor>& raster,
    const std::vector<torch::Tensor>& mlp,
    int64_t width, int64_t height, int64_t blocks, int64_t mode, bool profile);
std::vector<int64_t> gsplat_mixed_resources();
std::vector<std::vector<int64_t>> gsplat_standalone_resources();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gsplat_mixed_forward,
          pybind11::arg("raster"), pybind11::arg("mlp"),
          pybind11::arg("width"), pybind11::arg("height"),
          pybind11::arg("blocks") = 0, pybind11::arg("mode") = 2,
          pybind11::arg("profile") = false);
    m.def("resources", &gsplat_mixed_resources);
    m.def("standalone_resources", &gsplat_standalone_resources);
}
