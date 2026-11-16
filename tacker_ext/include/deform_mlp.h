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

// One-kernel experimental path with FP16-input/FP32-accumulate WMMA tails.
// packed_tail_weights has shape [112,128]: each of the five output groups is
// independently padded to a multiple of 16 rows (16/16/16/16/48).
std::vector<torch::Tensor> deform_mlp_tensor_tail_v3_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    int64_t persistent_blocks);

std::vector<torch::Tensor> deform_mlp_tensor_tail_out_v3_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t persistent_blocks);

// v4 preserves v3 and pads each compact FP16 activation row from 128 to 136
// elements before the Tensor-Core tail, targeting shared-load bank conflicts.
std::vector<torch::Tensor> deform_mlp_tensor_tail_v4_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    int64_t persistent_blocks);

std::vector<torch::Tensor> deform_mlp_tensor_tail_out_v4_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
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
