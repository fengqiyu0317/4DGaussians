#pragma once

#include <torch/extension.h>

torch::Tensor head_linear_solo_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias);

torch::Tensor head_linear_solo_out_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& output);

torch::Tensor head_linear_gptb_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t persistent_blocks);

torch::Tensor head_linear_gptb_out_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& output,
    int64_t persistent_blocks);
