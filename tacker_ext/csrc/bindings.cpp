#include <torch/extension.h>

#include <pybind11/stl.h>

#include <string>

#include "head_linear.h"
#include "head_linear_v2.h"

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

pybind11::dict TackerHeadCapabilitiesV2() {
    pybind11::dict capabilities;
    capabilities["abi_version"] = 2;
    capabilities["manifest"] = "abi/head_linear_v2.json";
    capabilities["sm_target"] = "sm_86";
    capabilities["head_features"] = 128;
    capabilities["max_head_tasks"] = 5;
    capabilities["worker_group_threads"] = 128;
    capabilities["max_worker_groups"] = 5;
    capabilities["max_backend_threads"] = 640;
    capabilities["max_mixed_cta_threads"] = 896;
    capabilities["resource_query"] = "tacker_resources_v2";

    pybind11::dict symbols;
    symbols["multi_solo"] = "tacker_head_linear_multi_solo_v2";
    symbols["multi_gptb"] = "tacker_head_linear_multi_gptb_v2";
    symbols["packed_gptb"] = "tacker_head_linear_packed_gptb_v2";
    symbols["whole_head_gptb"] = "tacker_whole_head_gptb_v2";
    capabilities["global_kernel_symbols"] = symbols;

    pybind11::dict adapters;
    adapters["multi"] = "tacker_4dgs::head_linear_multi_gptb_device";
    adapters["packed"] = "tacker_4dgs::head_linear_packed_gptb_device";
    adapters["whole_head"] = "tacker_4dgs::whole_head_multi_gptb_device";
    capabilities["device_adapters"] = adapters;

    capabilities["supported_head_roles"] = pybind11::make_tuple(
        "position", "scale", "rotation", "opacity", "sh");
    capabilities["supported_task_counts"] = pybind11::make_tuple(1, 2, 3, 4, 5);
    capabilities["supported_worker_groups"] =
        pybind11::make_tuple(1, 2, 3, 4, 5);
    capabilities["first_linear_named_barrier_ids"] = pybind11::tuple();
    capabilities["first_linear_dynamic_shared_memory_bytes"] = 0;

    pybind11::dict whole_head;
    whole_head["operation"] =
        "Linear(128,128,float16) -> ReLU -> Linear(128,O,float32)";
    whole_head["tail_features_min"] = 1;
    whole_head["tail_features_max"] = 128;
    whole_head["scratch_bytes_per_worker_group"] = 512;
    whole_head["named_barriers_per_worker_group"] = 1;
    capabilities["whole_head"] = whole_head;

    pybind11::list variants;
    const char* roles[] = {"position", "scale", "rotation", "opacity", "sh"};
    for (const char* role : roles) {
        pybind11::dict variant;
        variant["family"] = "C1";
        variant["variant_id"] = std::string("c1_") + role + "_first_linear";
        variant["task_count"] = 1;
        variant["worker_groups"] = 1;
        variant["symbol"] = "tacker_head_linear_multi_gptb_v2";
        variants.append(variant);
    }
    pybind11::dict c2;
    c2["family"] = "C2";
    c2["variant_id"] = "c2_position_scale_first_linear_wg2";
    c2["task_count"] = 2;
    c2["worker_groups"] = 2;
    c2["symbol"] = "tacker_head_linear_multi_gptb_v2";
    variants.append(c2);
    capabilities["canonical_variants"] = variants;
    return capabilities;
}

pybind11::dict TackerResourcesV2() {
    pybind11::dict result;
    result["abi_version"] = 2;
    pybind11::dict kernels;
    for (const auto& resource : tacker_kernel_resources_v2_cuda()) {
        pybind11::dict values;
        values["registers_per_thread"] = resource.registers_per_thread;
        values["static_shared_memory_bytes"] =
            resource.static_shared_memory_bytes;
        values["local_memory_bytes"] = resource.local_memory_bytes;
        values["max_threads_per_block"] = resource.max_threads_per_block;
        values["ptx_version"] = resource.ptx_version;
        values["binary_version"] = resource.binary_version;
        values["worker_group_threads"] = resource.worker_group_threads;
        values["active_blocks_per_sm"] = resource.active_blocks_per_sm;
        kernels[pybind11::str(resource.symbol)] = values;
    }
    result["kernels"] = kernels;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "tacker_capabilities",
        &TackerHeadCapabilities,
        "Return the compiled 4DGS head kernel ABI contract");
    module.def(
        "tacker_capabilities_v2",
        &TackerHeadCapabilitiesV2,
        "Return the compiled multi/packed/whole-head ABI v2 contract");
    module.def(
        "tacker_resources_v2",
        &TackerResourcesV2,
        "Return CUDA function resource attributes for ABI v2 kernels");
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
    module.def(
        "head_linear_multi_solo",
        &head_linear_multi_solo_cuda,
        "Run one to five independent first-linear heads in one launch",
        pybind11::arg("inputs"),
        pybind11::arg("weights"),
        pybind11::arg("biases"),
        pybind11::arg("worker_groups") = 1);
    module.def(
        "head_linear_multi_solo_out",
        &head_linear_multi_solo_out_cuda,
        "Run one to five independent first-linear heads into owned outputs",
        pybind11::arg("inputs"),
        pybind11::arg("weights"),
        pybind11::arg("biases"),
        pybind11::arg("outputs"),
        pybind11::arg("worker_groups") = 1);
    module.def(
        "head_linear_multi_gptb",
        &head_linear_multi_gptb_cuda,
        "Run one to five independent first-linear heads with persistent blocks",
        pybind11::arg("inputs"),
        pybind11::arg("weights"),
        pybind11::arg("biases"),
        pybind11::arg("worker_groups") = 1,
        pybind11::arg("persistent_blocks") = 0);
    module.def(
        "head_linear_multi_gptb_out",
        &head_linear_multi_gptb_out_cuda,
        "Persistent multi-head first-linear launch into owned outputs",
        pybind11::arg("inputs"),
        pybind11::arg("weights"),
        pybind11::arg("biases"),
        pybind11::arg("outputs"),
        pybind11::arg("worker_groups") = 1,
        pybind11::arg("persistent_blocks") = 0);
    module.def(
        "head_linear_packed_gptb",
        &head_linear_packed_gptb_cuda,
        "Persistent packed shared-input first-linear heads",
        pybind11::arg("input"),
        pybind11::arg("weights"),
        pybind11::arg("biases"),
        pybind11::arg("worker_groups") = 1,
        pybind11::arg("persistent_blocks") = 0);
    module.def(
        "head_linear_packed_gptb_out",
        &head_linear_packed_gptb_out_cuda,
        "Persistent packed shared-input heads into owned output",
        pybind11::arg("input"),
        pybind11::arg("weights"),
        pybind11::arg("biases"),
        pybind11::arg("output"),
        pybind11::arg("worker_groups") = 1,
        pybind11::arg("persistent_blocks") = 0);
    module.def(
        "whole_head_gptb",
        &whole_head_gptb_cuda,
        "Persistent Linear-ReLU-tail whole-head adapter",
        pybind11::arg("input"),
        pybind11::arg("first_weight"),
        pybind11::arg("first_bias"),
        pybind11::arg("tail_weight"),
        pybind11::arg("tail_bias"),
        pybind11::arg("persistent_blocks") = 0);
}
