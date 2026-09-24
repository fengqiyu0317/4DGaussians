#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> deform_mlp_delta_v1_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights,
    const torch::Tensor& tail_biases,
    int64_t persistent_blocks);

std::vector<torch::Tensor> deform_mlp_delta_out_v1_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights,
    const torch::Tensor& tail_biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t persistent_blocks);

// Six-launch experimental path.  The first kernel materializes the common
// FP16 backbone activation [N,128].  Five head-specialized kernels then launch
// eight independent slice blocks per 128-row group, serially traverse 16-row
// WMMA micro-tiles, and write disjoint tail partials as [8,N,59].  The heads
// run on independent pooled streams and rejoin the caller's current stream
// before the caller-owned final slice reduction and tail-bias add.
std::vector<torch::Tensor> deform_mlp_sliced_v2_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights);

std::vector<torch::Tensor> deform_mlp_sliced_out_v2_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights,
    const torch::Tensor& activated,
    const torch::Tensor& partials);
