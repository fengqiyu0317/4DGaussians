#pragma once

#include <torch/extension.h>

#include <cstddef>
#include <string>
#include <vector>

struct TackerKernelResourceV2 {
    std::string symbol;
    int registers_per_thread;
    std::size_t static_shared_memory_bytes;
    std::size_t local_memory_bytes;
    int max_threads_per_block;
    int ptx_version;
    int binary_version;
    std::vector<int> worker_group_threads;
    std::vector<int> active_blocks_per_sm;
};

std::vector<torch::Tensor> head_linear_multi_solo_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    int64_t worker_groups);

std::vector<torch::Tensor> head_linear_multi_solo_out_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t worker_groups);

std::vector<torch::Tensor> head_linear_multi_gptb_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    int64_t worker_groups,
    int64_t persistent_blocks);

std::vector<torch::Tensor> head_linear_multi_gptb_out_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t worker_groups,
    int64_t persistent_blocks);

torch::Tensor head_linear_packed_gptb_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& biases,
    int64_t worker_groups,
    int64_t persistent_blocks);

torch::Tensor head_linear_packed_gptb_out_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& biases,
    const torch::Tensor& output,
    int64_t worker_groups,
    int64_t persistent_blocks);

torch::Tensor whole_head_gptb_cuda(
    const torch::Tensor& input,
    const torch::Tensor& first_weight,
    const torch::Tensor& first_bias,
    const torch::Tensor& tail_weight,
    const torch::Tensor& tail_bias,
    int64_t persistent_blocks);

std::vector<TackerKernelResourceV2> tacker_kernel_resources_v2_cuda();
