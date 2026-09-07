#include <torch/extension.h>

#include "head_linear.h"

namespace {

pybind11::dict TackerHeadCapabilities() {
    pybind11::dict capabilities;
    capabilities["abi_version"] = 1;
    capabilities["sm_target"] = "sm_86";
    capabilities["head_features"] = 128;
    capabilities["block_threads"] = 128;
    capabilities["input_dtype"] = "float16";
    capabilities["weight_dtype"] = "float16";
    capabilities["bias_dtype"] = "float32";
    capabilities["accumulation_dtype"] = "float32";
    capabilities["output_dtype"] = "float32";
    capabilities["solo_symbol"] = "tacker_head_linear_solo_v1";
    capabilities["gptb_symbol"] = "tacker_head_linear_gptb_v1";
    return capabilities;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "tacker_capabilities",
        &TackerHeadCapabilities,
        "Return the compiled 4DGS head kernel ABI contract");
    module.def(
        "head_linear_solo",
        &head_linear_solo_cuda,
        "4DGS 128x128 head Linear (one block per tile)");
    module.def(
        "head_linear_solo_out",
        &head_linear_solo_out_cuda,
        "4DGS 128x128 head Linear into caller-owned output",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("output"));
    module.def(
        "head_linear_gptb",
        &head_linear_gptb_cuda,
        "4DGS 128x128 head Linear (persistent GPTB blocks)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("persistent_blocks") = 0);
    module.def(
        "head_linear_gptb_out",
        &head_linear_gptb_out_cuda,
        "4DGS 128x128 persistent Linear into caller-owned output",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("output"),
        pybind11::arg("persistent_blocks") = 0);
}
