#pragma once

#include <torch/extension.h>
#include <vector>

torch::Tensor hexplane_sample_plane_cuda(
    const torch::Tensor& coords, const torch::Tensor& plane,
    int64_t axis_x, int64_t axis_y, bool probe_512);
torch::Tensor hexplane_sample_plane_out_cuda(
    const torch::Tensor& coords, const torch::Tensor& plane,
    const torch::Tensor& output, int64_t axis_x, int64_t axis_y, bool probe_512);

// [threads, registers/thread, local bytes/thread, static shared bytes,
//  active blocks/SM, binary version] for each compiled wrapper.
std::vector<std::vector<int64_t>> hexplane_sample_resources_cuda();

// B: one launch for 1..12 planes, one 256-point tile per physical CTA.
std::vector<torch::Tensor> hexplane_sample_planes_cuda(
    const torch::Tensor& coords, const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes);
std::vector<torch::Tensor> hexplane_sample_planes_out_cuda(
    const torch::Tensor& coords, const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes,
    const std::vector<torch::Tensor>& outputs);
std::vector<int64_t> hexplane_sample_batch_resources_cuda();

// Twelve [N,16] column-major samples -> one contiguous [N,32] feature tensor.
torch::Tensor hexplane_reduce_features_cuda(
    const std::vector<torch::Tensor>& samples, int64_t threads);
torch::Tensor hexplane_reduce_features_out_cuda(
    const std::vector<torch::Tensor>& samples, const torch::Tensor& output,
    int64_t threads);
std::vector<std::vector<int64_t>> hexplane_reduce_resources_cuda();
