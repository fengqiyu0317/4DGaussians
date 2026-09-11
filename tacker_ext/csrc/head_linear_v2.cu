#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "head_linear_v2.h"
#include "head_linear_v2_device.cuh"
#include "head_linear_v2_kernels.cuh"

namespace {

using tacker_4dgs::HeadLinearTaskV2;
using tacker_4dgs::WholeHeadTaskV2;
using tacker_4dgs::kHeadFeatures;
using tacker_4dgs::kHeadThreads;
using tacker_4dgs::kMaxHeadTasksV2;

constexpr uintptr_t kV2Alignment = 32;

static_assert(sizeof(void*) == 8, "head ABI v2 requires 64-bit device pointers");
static_assert(offsetof(HeadLinearTaskV2, input) == 0, "v2 task layout changed");
static_assert(offsetof(HeadLinearTaskV2, weight) == 8, "v2 task layout changed");
static_assert(offsetof(HeadLinearTaskV2, bias) == 16, "v2 task layout changed");
static_assert(offsetof(HeadLinearTaskV2, output) == 24, "v2 task layout changed");
static_assert(offsetof(HeadLinearTaskV2, rows) == 32, "v2 task layout changed");
static_assert(sizeof(HeadLinearTaskV2) == 40, "v2 task size changed");
static_assert(sizeof(WholeHeadTaskV2) == 56, "v2 whole-head task size changed");

bool is_aligned_v2(const torch::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kV2Alignment == 0;
}

uint64_t tensor_byte_count_v2(const torch::Tensor& tensor) {
    const uint64_t elements = static_cast<uint64_t>(tensor.numel());
    const uint64_t element_size = static_cast<uint64_t>(tensor.element_size());
    TORCH_CHECK(
        element_size == 0 ||
            elements <= std::numeric_limits<uint64_t>::max() / element_size,
        "tensor byte size overflows uint64 range");
    return elements * element_size;
}

bool byte_ranges_overlap_v2(
    const torch::Tensor& left,
    const torch::Tensor& right) {
    if (left.numel() == 0 || right.numel() == 0) {
        return false;
    }
    const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
    const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
    const uint64_t left_bytes = tensor_byte_count_v2(left);
    const uint64_t right_bytes = tensor_byte_count_v2(right);
    if (left_begin <= right_begin) {
        return static_cast<uint64_t>(right_begin - left_begin) < left_bytes;
    }
    return static_cast<uint64_t>(left_begin - right_begin) < right_bytes;
}

void check_inference_v2() {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "head ABI v2 is inference-only; call it under torch.no_grad()");
}

void check_worker_groups_v2(int64_t task_count, int64_t worker_groups) {
    TORCH_CHECK(task_count >= 1, "head task sequence must not be empty");
    TORCH_CHECK(
        task_count <= kMaxHeadTasksV2,
        "head task sequence supports at most 5 heads");
    TORCH_CHECK(worker_groups >= 1, "worker_groups must be >= 1");
    TORCH_CHECK(
        worker_groups <= task_count,
        "worker_groups must not exceed head task count");
}

void check_first_linear_tensor_v2(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t task_index) {
    const std::string prefix =
        "head task " + std::to_string(task_index) + " ";
    TORCH_CHECK(input.is_cuda(), prefix, "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), prefix, "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), prefix, "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), prefix, "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), prefix, "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), prefix, "bias must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf, prefix, "input must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, prefix, "weight must be float16");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, prefix, "bias must be float32");
    TORCH_CHECK(
        input.dim() == 2 && input.size(1) == kHeadFeatures,
        prefix,
        "input must have shape [N, 128]");
    TORCH_CHECK(
        weight.dim() == 2 && weight.size(0) == kHeadFeatures &&
            weight.size(1) == kHeadFeatures,
        prefix,
        "weight must have shape [128, 128]");
    TORCH_CHECK(
        bias.dim() == 1 && bias.size(0) == kHeadFeatures,
        prefix,
        "bias must have shape [128]");
    TORCH_CHECK(
        input.device() == weight.device() && input.device() == bias.device(),
        prefix,
        "input, weight, and bias must be on the same CUDA device");
    TORCH_CHECK(
        input.size(0) <= static_cast<int64_t>(INT32_MAX),
        prefix,
        "input row count exceeds the int32 kernel ABI");
    TORCH_CHECK(is_aligned_v2(input), prefix, "input data pointer must be 32-byte aligned");
    TORCH_CHECK(is_aligned_v2(weight), prefix, "weight data pointer must be 32-byte aligned");
    TORCH_CHECK(is_aligned_v2(bias), prefix, "bias data pointer must be 32-byte aligned");
}

void check_first_linear_output_v2(
    const torch::Tensor& input,
    const torch::Tensor& output,
    int64_t task_index) {
    const std::string prefix =
        "head task " + std::to_string(task_index) + " ";
    TORCH_CHECK(output.is_cuda(), prefix, "output must be a CUDA tensor");
    TORCH_CHECK(output.is_contiguous(), prefix, "output must be contiguous");
    TORCH_CHECK(output.scalar_type() == at::kFloat, prefix, "output must be float32");
    TORCH_CHECK(
        output.dim() == 2 && output.size(0) == input.size(0) &&
            output.size(1) == kHeadFeatures,
        prefix,
        "output must have shape [N, 128]");
    TORCH_CHECK(
        output.device() == input.device(),
        prefix,
        "output must be on the same CUDA device as input");
    TORCH_CHECK(is_aligned_v2(output), prefix, "output data pointer must be 32-byte aligned");
}

void check_multi_read_sequences_v2(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    int64_t worker_groups) {
    check_inference_v2();
    check_worker_groups_v2(inputs.size(), worker_groups);
    TORCH_CHECK(
        weights.size() == inputs.size() && biases.size() == inputs.size(),
        "inputs, weights, and biases must have equal sequence lengths");
    const auto device = inputs.front().device();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        check_first_linear_tensor_v2(
            inputs[index], weights[index], biases[index], index);
        TORCH_CHECK(
            inputs[index].device() == device,
            "all head tasks must be on the same CUDA device");
    }
}

void check_multi_sequences_v2(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t worker_groups) {
    check_multi_read_sequences_v2(inputs, weights, biases, worker_groups);
    TORCH_CHECK(
        outputs.size() == inputs.size(),
        "outputs must have the same sequence length as inputs");
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        check_first_linear_output_v2(inputs[index], outputs[index], index);
    }

    // Outputs execute concurrently and therefore must be disjoint from every
    // read operand and from each other.  Inputs and parameters may be shared.
    for (std::size_t output_index = 0; output_index < outputs.size(); ++output_index) {
        for (std::size_t task_index = 0; task_index < inputs.size(); ++task_index) {
            TORCH_CHECK(
                !byte_ranges_overlap_v2(outputs[output_index], inputs[task_index]),
                "head outputs must not overlap any input storage");
            TORCH_CHECK(
                !byte_ranges_overlap_v2(outputs[output_index], weights[task_index]),
                "head outputs must not overlap any weight storage");
            TORCH_CHECK(
                !byte_ranges_overlap_v2(outputs[output_index], biases[task_index]),
                "head outputs must not overlap any bias storage");
        }
        for (std::size_t other = output_index + 1; other < outputs.size(); ++other) {
            TORCH_CHECK(
                !byte_ranges_overlap_v2(outputs[output_index], outputs[other]),
                "head output storages must not overlap each other");
        }
    }
}

HeadLinearTaskV2 make_task_v2(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& output) {
    HeadLinearTaskV2 task;
    task.input = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
    task.weight = reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
    task.bias = bias.data_ptr<float>();
    task.output = output.data_ptr<float>();
    task.rows = static_cast<int>(input.size(0));
    return task;
}

HeadLinearTaskV2 empty_task_v2() {
    HeadLinearTaskV2 task = {nullptr, nullptr, nullptr, nullptr, 0};
    return task;
}

int max_first_linear_blocks_v2(const std::vector<torch::Tensor>& inputs) {
    int maximum = 0;
    for (const auto& input : inputs) {
        const int64_t grid_x =
            (input.size(0) + tacker_4dgs::kRowsPerTile - 1) /
            tacker_4dgs::kRowsPerTile;
        const int64_t blocks = grid_x * 2;
        TORCH_CHECK(blocks <= INT32_MAX, "head logical grid exceeds int32 ABI");
        maximum = std::max(maximum, static_cast<int>(blocks));
    }
    return maximum;
}

int resolve_physical_blocks_v2(
    int64_t requested,
    int logical_blocks) {
    TORCH_CHECK(requested >= 0, "persistent_blocks must be >= 0");
    TORCH_CHECK(requested <= INT32_MAX, "persistent_blocks exceeds int32 range");
    if (logical_blocks == 0) {
        return 0;
    }
    int physical = static_cast<int>(requested);
    if (physical == 0) {
        physical = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    }
    return std::max(1, std::min(physical, logical_blocks));
}

std::vector<torch::Tensor> allocate_multi_outputs_v2(
    const std::vector<torch::Tensor>& inputs) {
    std::vector<torch::Tensor> outputs;
    outputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        outputs.push_back(torch::empty(
            {input.size(0), kHeadFeatures},
            input.options().dtype(torch::kFloat32)));
    }
    return outputs;
}

void launch_multi_v2(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t worker_groups,
    int64_t persistent_blocks,
    bool persistent) {
    check_multi_sequences_v2(inputs, weights, biases, outputs, worker_groups);
    const c10::cuda::CUDAGuard device_guard(inputs.front().device());
    const int logical_end = max_first_linear_blocks_v2(inputs);
    if (logical_end == 0) {
        return;
    }

    HeadLinearTaskV2 tasks[kMaxHeadTasksV2];
    for (int index = 0; index < kMaxHeadTasksV2; ++index) {
        tasks[index] = empty_task_v2();
    }
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        tasks[index] = make_task_v2(
            inputs[index], weights[index], biases[index], outputs[index]);
    }

    int physical_blocks = logical_end;
    if (persistent) {
        physical_blocks = resolve_physical_blocks_v2(
            persistent_blocks, logical_end);
    }
    const int threads = static_cast<int>(worker_groups) * kHeadThreads;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (persistent) {
        tacker_head_linear_multi_gptb_v2<<<physical_blocks, threads, 0, stream>>>(
            tasks[0], tasks[1], tasks[2], tasks[3], tasks[4],
            static_cast<int>(inputs.size()),
            static_cast<int>(worker_groups),
            0,
            physical_blocks,
            logical_end);
    } else {
        tacker_head_linear_multi_solo_v2<<<physical_blocks, threads, 0, stream>>>(
            tasks[0], tasks[1], tasks[2], tasks[3], tasks[4],
            static_cast<int>(inputs.size()),
            static_cast<int>(worker_groups),
            logical_end);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TackerKernelResourceV2 read_kernel_resource_v2(
    const char* symbol,
    const void* function,
    int max_worker_groups) {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, function));
    TackerKernelResourceV2 result;
    result.symbol = symbol;
    result.registers_per_thread = attributes.numRegs;
    result.static_shared_memory_bytes = attributes.sharedSizeBytes;
    result.local_memory_bytes = attributes.localSizeBytes;
    result.max_threads_per_block = attributes.maxThreadsPerBlock;
    result.ptx_version = attributes.ptxVersion;
    result.binary_version = attributes.binaryVersion;
    for (int worker_groups = 1; worker_groups <= max_worker_groups;
         ++worker_groups) {
        const int threads = worker_groups * kHeadThreads;
        if (threads > attributes.maxThreadsPerBlock) {
            break;
        }
        int active_blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, function, threads, 0));
        result.worker_group_threads.push_back(threads);
        result.active_blocks_per_sm.push_back(active_blocks);
    }
    return result;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(640)
tacker_head_linear_multi_solo_v2(
    HeadLinearTaskV2 task0,
    HeadLinearTaskV2 task1,
    HeadLinearTaskV2 task2,
    HeadLinearTaskV2 task3,
    HeadLinearTaskV2 task4,
    int task_count,
    int worker_groups,
    int logical_end) {
    const HeadLinearTaskV2 tasks[tacker_4dgs::kMaxHeadTasksV2] = {
        task0, task1, task2, task3, task4};
    tacker_4dgs::head_linear_multi_gptb_device(
        tasks,
        task_count,
        worker_groups,
        0,
        static_cast<int>(gridDim.x),
        logical_end,
        0);
}

extern "C" __global__ void __launch_bounds__(640)
tacker_head_linear_multi_gptb_v2(
    HeadLinearTaskV2 task0,
    HeadLinearTaskV2 task1,
    HeadLinearTaskV2 task2,
    HeadLinearTaskV2 task3,
    HeadLinearTaskV2 task4,
    int task_count,
    int worker_groups,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos) {
    const HeadLinearTaskV2 tasks[tacker_4dgs::kMaxHeadTasksV2] = {
        task0, task1, task2, task3, task4};
    tacker_4dgs::head_linear_multi_gptb_device(
        tasks,
        task_count,
        worker_groups,
        ptb_start_block_pos,
        ptb_iter_block_step,
        ptb_end_block_pos,
        0);
}

extern "C" __global__ void __launch_bounds__(640)
tacker_head_linear_packed_gptb_v2(
    const half* input,
    const half* packed_weights,
    const float* packed_biases,
    float* packed_outputs,
    int rows,
    int head_count,
    int worker_groups,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos) {
    tacker_4dgs::head_linear_packed_gptb_device(
        input,
        packed_weights,
        packed_biases,
        packed_outputs,
        rows,
        head_count,
        worker_groups,
        ptb_start_block_pos,
        ptb_iter_block_step,
        ptb_end_block_pos,
        0);
}

extern "C" __global__ void __launch_bounds__(128)
tacker_whole_head_gptb_v2(
    WholeHeadTaskV2 task,
    int ptb_start_row,
    int ptb_iter_row_step,
    int ptb_end_row) {
    __shared__ float hidden[tacker_4dgs::kWholeHeadScratchFloatsPerGroupV2];
    const WholeHeadTaskV2 tasks[1] = {task};
    tacker_4dgs::whole_head_multi_gptb_device(
        tasks,
        1,
        1,
        ptb_start_row,
        ptb_iter_row_step,
        ptb_end_row,
        0,
        hidden,
        0);
}

std::vector<torch::Tensor> head_linear_multi_solo_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    int64_t worker_groups) {
    check_inference_v2();
    check_worker_groups_v2(inputs.size(), worker_groups);
    TORCH_CHECK(
        inputs.size() == weights.size() && inputs.size() == biases.size(),
        "inputs, weights, and biases must have equal sequence lengths");
    check_multi_read_sequences_v2(inputs, weights, biases, worker_groups);
    auto outputs = allocate_multi_outputs_v2(inputs);
    launch_multi_v2(
        inputs, weights, biases, outputs, worker_groups, 0, false);
    return outputs;
}

std::vector<torch::Tensor> head_linear_multi_solo_out_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t worker_groups) {
    launch_multi_v2(
        inputs, weights, biases, outputs, worker_groups, 0, false);
    return outputs;
}

std::vector<torch::Tensor> head_linear_multi_gptb_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    int64_t worker_groups,
    int64_t persistent_blocks) {
    check_inference_v2();
    check_worker_groups_v2(inputs.size(), worker_groups);
    TORCH_CHECK(
        inputs.size() == weights.size() && inputs.size() == biases.size(),
        "inputs, weights, and biases must have equal sequence lengths");
    check_multi_read_sequences_v2(inputs, weights, biases, worker_groups);
    auto outputs = allocate_multi_outputs_v2(inputs);
    launch_multi_v2(
        inputs,
        weights,
        biases,
        outputs,
        worker_groups,
        persistent_blocks,
        true);
    return outputs;
}

std::vector<torch::Tensor> head_linear_multi_gptb_out_cuda(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t worker_groups,
    int64_t persistent_blocks) {
    launch_multi_v2(
        inputs,
        weights,
        biases,
        outputs,
        worker_groups,
        persistent_blocks,
        true);
    return outputs;
}

torch::Tensor head_linear_packed_gptb_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& biases,
    int64_t worker_groups,
    int64_t persistent_blocks) {
    check_inference_v2();
    TORCH_CHECK(
        weights.dim() == 3,
        "packed weights must have shape [H, 128, 128]");
    auto output = torch::empty(
        {weights.size(0), input.size(0), kHeadFeatures},
        input.options().dtype(torch::kFloat32));
    return head_linear_packed_gptb_out_cuda(
        input, weights, biases, output, worker_groups, persistent_blocks);
}

torch::Tensor head_linear_packed_gptb_out_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& biases,
    const torch::Tensor& output,
    int64_t worker_groups,
    int64_t persistent_blocks) {
    check_inference_v2();
    TORCH_CHECK(weights.dim() == 3, "packed weights must have shape [H, 128, 128]");
    const int64_t head_count = weights.size(0);
    check_worker_groups_v2(head_count, worker_groups);
    TORCH_CHECK(input.is_cuda(), "packed input must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "packed weights must be a CUDA tensor");
    TORCH_CHECK(biases.is_cuda(), "packed biases must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "packed output must be a CUDA tensor");
    TORCH_CHECK(
        input.is_contiguous() && weights.is_contiguous() &&
            biases.is_contiguous() && output.is_contiguous(),
        "packed tensors must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "packed input must be float16");
    TORCH_CHECK(weights.scalar_type() == at::kHalf, "packed weights must be float16");
    TORCH_CHECK(biases.scalar_type() == at::kFloat, "packed biases must be float32");
    TORCH_CHECK(output.scalar_type() == at::kFloat, "packed output must be float32");
    TORCH_CHECK(
        input.dim() == 2 && input.size(1) == kHeadFeatures,
        "packed input must have shape [N, 128]");
    TORCH_CHECK(
        weights.size(1) == kHeadFeatures && weights.size(2) == kHeadFeatures,
        "packed weights must have shape [H, 128, 128]");
    TORCH_CHECK(
        biases.dim() == 2 && biases.size(0) == head_count &&
            biases.size(1) == kHeadFeatures,
        "packed biases must have shape [H, 128]");
    TORCH_CHECK(
        output.dim() == 3 && output.size(0) == head_count &&
            output.size(1) == input.size(0) &&
            output.size(2) == kHeadFeatures,
        "packed output must have shape [H, N, 128]");
    TORCH_CHECK(
        input.device() == weights.device() && input.device() == biases.device() &&
            input.device() == output.device(),
        "packed tensors must be on the same CUDA device");
    TORCH_CHECK(
        input.size(0) <= static_cast<int64_t>(INT32_MAX),
        "packed row count exceeds int32 ABI");
    TORCH_CHECK(
        is_aligned_v2(input) && is_aligned_v2(weights) &&
            is_aligned_v2(biases) && is_aligned_v2(output),
        "packed tensor data pointers must be 32-byte aligned");
    TORCH_CHECK(
        !byte_ranges_overlap_v2(output, input) &&
            !byte_ranges_overlap_v2(output, weights) &&
            !byte_ranges_overlap_v2(output, biases),
        "packed output storage must not alias an input or parameter");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) {
        return output;
    }
    const int logical_blocks =
        static_cast<int>((input.size(0) + tacker_4dgs::kRowsPerTile - 1) /
                         tacker_4dgs::kRowsPerTile) * 2;
    const int physical_blocks =
        resolve_physical_blocks_v2(persistent_blocks, logical_blocks);
    const int threads = static_cast<int>(worker_groups) * kHeadThreads;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tacker_head_linear_packed_gptb_v2<<<physical_blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weights.data_ptr<at::Half>()),
        biases.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        static_cast<int>(head_count),
        static_cast<int>(worker_groups),
        0,
        physical_blocks,
        logical_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor whole_head_gptb_cuda(
    const torch::Tensor& input,
    const torch::Tensor& first_weight,
    const torch::Tensor& first_bias,
    const torch::Tensor& tail_weight,
    const torch::Tensor& tail_bias,
    int64_t persistent_blocks) {
    check_inference_v2();
    check_first_linear_tensor_v2(input, first_weight, first_bias, 0);
    TORCH_CHECK(tail_weight.is_cuda(), "tail weight must be a CUDA tensor");
    TORCH_CHECK(tail_bias.is_cuda(), "tail bias must be a CUDA tensor");
    TORCH_CHECK(tail_weight.is_contiguous(), "tail weight must be contiguous");
    TORCH_CHECK(tail_bias.is_contiguous(), "tail bias must be contiguous");
    TORCH_CHECK(tail_weight.scalar_type() == at::kFloat, "tail weight must be float32");
    TORCH_CHECK(tail_bias.scalar_type() == at::kFloat, "tail bias must be float32");
    TORCH_CHECK(
        tail_weight.dim() == 2 && tail_weight.size(1) == kHeadFeatures &&
            tail_weight.size(0) >= 1 &&
            tail_weight.size(0) <= tacker_4dgs::kMaxTailFeaturesV2,
        "tail weight must have shape [O, 128] with 1 <= O <= 128");
    TORCH_CHECK(
        tail_bias.dim() == 1 && tail_bias.size(0) == tail_weight.size(0),
        "tail bias must have shape [O]");
    TORCH_CHECK(
        input.device() == tail_weight.device() &&
            input.device() == tail_bias.device(),
        "whole-head tensors must be on the same CUDA device");
    TORCH_CHECK(
        is_aligned_v2(tail_weight) && is_aligned_v2(tail_bias),
        "whole-head parameter pointers must be 32-byte aligned");

    auto output = torch::empty(
        {input.size(0), tail_weight.size(0)},
        input.options().dtype(torch::kFloat32));
    TORCH_CHECK(is_aligned_v2(output), "whole-head output must be 32-byte aligned");
    TORCH_CHECK(
        !byte_ranges_overlap_v2(output, input) &&
            !byte_ranges_overlap_v2(output, first_weight) &&
            !byte_ranges_overlap_v2(output, first_bias) &&
            !byte_ranges_overlap_v2(output, tail_weight) &&
            !byte_ranges_overlap_v2(output, tail_bias),
        "whole-head output storage must not alias an input or parameter");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) {
        return output;
    }
    const int physical_blocks =
        resolve_physical_blocks_v2(persistent_blocks, rows);
    WholeHeadTaskV2 task;
    task.input = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
    task.first_weight =
        reinterpret_cast<const half*>(first_weight.data_ptr<at::Half>());
    task.first_bias = first_bias.data_ptr<float>();
    task.tail_weight = tail_weight.data_ptr<float>();
    task.tail_bias = tail_bias.data_ptr<float>();
    task.output = output.data_ptr<float>();
    task.rows = rows;
    task.tail_features = static_cast<int>(tail_weight.size(0));

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tacker_whole_head_gptb_v2<<<physical_blocks, kHeadThreads, 0, stream>>>(
        task, 0, physical_blocks, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<TackerKernelResourceV2> tacker_kernel_resources_v2_cuda() {
    std::vector<TackerKernelResourceV2> resources;
    resources.push_back(read_kernel_resource_v2(
        "tacker_head_linear_multi_solo_v2",
        reinterpret_cast<const void*>(tacker_head_linear_multi_solo_v2),
        kMaxHeadTasksV2));
    resources.push_back(read_kernel_resource_v2(
        "tacker_head_linear_multi_gptb_v2",
        reinterpret_cast<const void*>(tacker_head_linear_multi_gptb_v2),
        kMaxHeadTasksV2));
    resources.push_back(read_kernel_resource_v2(
        "tacker_head_linear_packed_gptb_v2",
        reinterpret_cast<const void*>(tacker_head_linear_packed_gptb_v2),
        kMaxHeadTasksV2));
    resources.push_back(read_kernel_resource_v2(
        "tacker_whole_head_gptb_v2",
        reinterpret_cast<const void*>(tacker_whole_head_gptb_v2),
        1));
    return resources;
}
