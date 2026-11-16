#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "config.h"
#include "deform_mlp_device.cuh"
#include "tacker_forward.cuh"
#include "tacker_mixed.h"

// Keep the body and launch ABI of tacker_mix_render_deform_mlp_v5. Timing is
// done by host-side CUDA Graph replay, so the CTA has no timing instrumentation.
extern "C" __global__ void __launch_bounds__(CudaRasterizer::Tacker::kMixedThreads)
synthetic_mix_render_deform_mlp_v5(
    const uint2* ranges,
    const uint32_t* point_list,
    int width,
    int height,
    const float2* points_xy_image,
    const float* features,
    const float* depths,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* background,
    float* out_color,
    float* out_depth,
    int raster_grid_x,
    int raster_grid_y,
    CudaRasterizer::MixedDeformMlpTaskV1 deform_task,
    int physical_blocks) {
    CudaRasterizer::Tacker::general_ptb_render<NUM_CHANNELS>(
        ranges, point_list, width, height, points_xy_image, features, depths,
        conic_opacity, final_T, n_contrib, background, out_color, out_depth,
        raster_grid_x, raster_grid_y, 0, physical_blocks,
        raster_grid_x * raster_grid_y, 0,
        CudaRasterizer::Tacker::kRasterBarrierId);

    __shared__ tacker_4dgs::DeformMlpParamsV1 shared_deform;
    __shared__ tacker_4dgs::DeformMlpScratchV1 deform_scratch;
    const int local_thread = static_cast<int>(threadIdx.x) -
        CudaRasterizer::Tacker::kHeadThreadBase;
    if (local_thread >= 0 && local_thread < tacker_4dgs::kDeformThreads) {
        if (local_thread == 0) {
            shared_deform.features = deform_task.features;
            shared_deform.backbone_weight = deform_task.backbone_weight;
            shared_deform.backbone_bias = deform_task.backbone_bias;
            shared_deform.first_weights = reinterpret_cast<const half*>(
                deform_task.first_weights);
            shared_deform.first_biases = deform_task.first_biases;
            shared_deform.tail_weights = deform_task.tail_weights;
            shared_deform.tail_biases = deform_task.tail_biases;
#pragma unroll
            for (int output = 0; output < tacker_4dgs::kDeformHeads; ++output) {
                shared_deform.outputs[output] = deform_task.outputs[output];
            }
            shared_deform.rows = deform_task.rows;
        }
        CudaRasterizer::Tacker::subgroup_sync(
            CudaRasterizer::Tacker::kDeformMlpBarrierIdV1,
            tacker_4dgs::kDeformThreads);
        tacker_4dgs::deform_mlp_gptb_device_v1(
            shared_deform, deform_scratch, static_cast<int>(blockIdx.x),
            physical_blocks, CudaRasterizer::Tacker::kHeadThreadBase,
            CudaRasterizer::Tacker::kDeformMlpBarrierIdV1);
    }
}

namespace {

constexpr int kMaxRasterTasks = 2;
constexpr int kMaxDeformTasks = 3;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif
#define CHECK_CUDA(call) check_cuda((call), #call)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t count) {
        if (data_ != nullptr) CHECK_CUDA(cudaFree(data_));
        count_ = std::max<size_t>(count, 1);
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }
    void zero() { CHECK_CUDA(cudaMemset(data_, 0, count_ * sizeof(T))); }
    void copy_from(const std::vector<T>& host) {
        if (host.size() > count_) {
            throw std::runtime_error("host copy exceeds device allocation");
        }
        CHECK_CUDA(cudaMemcpy(
            data_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

struct RasterBuffers {
    DeviceBuffer<uint2> ranges{kMaxRasterTasks};
    DeviceBuffer<uint32_t> point_list;
    DeviceBuffer<float2> points;
    DeviceBuffer<float> features;
    DeviceBuffer<float> depths;
    DeviceBuffer<float4> conics;
    DeviceBuffer<float> final_t{kMaxRasterTasks * BLOCK_X * BLOCK_Y};
    DeviceBuffer<uint32_t> contributors{kMaxRasterTasks * BLOCK_X * BLOCK_Y};
    DeviceBuffer<float> background{NUM_CHANNELS};
    DeviceBuffer<float> out_color{
        kMaxRasterTasks * BLOCK_X * BLOCK_Y * NUM_CHANNELS};
    DeviceBuffer<float> out_depth{kMaxRasterTasks * BLOCK_X * BLOCK_Y};

    RasterBuffers(int first_tile_gaussians, int second_tile_gaussians) {
        const int tile_gaussians[kMaxRasterTasks] = {
            first_tile_gaussians, second_tile_gaussians};
        const int total_gaussians = first_tile_gaussians + second_tile_gaussians;
        const int count = std::max(1, total_gaussians);
        point_list.allocate(count);
        points.allocate(count);
        features.allocate(static_cast<size_t>(count) * NUM_CHANNELS);
        depths.allocate(count);
        conics.allocate(count);

        std::vector<uint2> host_ranges(kMaxRasterTasks);
        host_ranges[0] = make_uint2(0, first_tile_gaussians);
        host_ranges[1] = make_uint2(first_tile_gaussians, total_gaussians);
        ranges.copy_from(host_ranges);

        std::vector<uint32_t> host_list(count);
        std::vector<float2> host_points(count);
        std::vector<float> host_features(
            static_cast<size_t>(count) * NUM_CHANNELS, 0.25f);
        std::vector<float> host_depths(count, 1.0f);
        std::vector<float4> host_conics(count);
        for (int index = 0; index < count; ++index) {
            host_list[index] = static_cast<uint32_t>(index);
            const int tile = index < tile_gaussians[0] ? 0 : 1;
            host_points[index] = make_float2(
                tile * BLOCK_X + 0.5f * (BLOCK_X - 1),
                0.5f * (BLOCK_Y - 1));
            // Every reference takes the blend path.  Even at K=1664,
            // transmittance remains above the renderer's early-out threshold.
            host_conics[index] = make_float4(0.0f, 0.0f, 0.0f, 0.005f);
        }
        point_list.copy_from(host_list);
        points.copy_from(host_points);
        features.copy_from(host_features);
        depths.copy_from(host_depths);
        conics.copy_from(host_conics);
        final_t.zero();
        contributors.zero();
        background.zero();
        out_color.zero();
        out_depth.zero();
    }
};

struct DeformBuffers {
    static constexpr int kMaxRows =
        kMaxDeformTasks * tacker_4dgs::kDeformRows;
    DeviceBuffer<float> features{kMaxRows * tacker_4dgs::kDeformInput};
    DeviceBuffer<float> backbone_weight{
        tacker_4dgs::kDeformInput * tacker_4dgs::kDeformWidth};
    DeviceBuffer<float> backbone_bias{tacker_4dgs::kDeformWidth};
    DeviceBuffer<half> first_weights{
        tacker_4dgs::kDeformHeads * tacker_4dgs::kDeformWidth *
        tacker_4dgs::kDeformWidth};
    DeviceBuffer<float> first_biases{
        tacker_4dgs::kDeformHeads * tacker_4dgs::kDeformWidth};
    DeviceBuffer<float> tail_weights{59 * tacker_4dgs::kDeformWidth};
    DeviceBuffer<float> tail_biases{59};
    DeviceBuffer<float> out_position{kMaxRows * 3};
    DeviceBuffer<float> out_scale{kMaxRows * 3};
    DeviceBuffer<float> out_rotation{kMaxRows * 4};
    DeviceBuffer<float> out_opacity{kMaxRows};
    DeviceBuffer<float> out_sh{kMaxRows * 48};

    DeformBuffers() {
        features.zero();
        backbone_weight.zero();
        backbone_bias.zero();
        first_weights.zero();
        first_biases.zero();
        tail_weights.zero();
        tail_biases.zero();
        out_position.zero();
        out_scale.zero();
        out_rotation.zero();
        out_opacity.zero();
        out_sh.zero();
    }

    CudaRasterizer::MixedDeformMlpTaskV1 task(int deform_tasks) {
        CudaRasterizer::MixedDeformMlpTaskV1 result{};
        result.features = features.get();
        result.backbone_weight = backbone_weight.get();
        result.backbone_bias = backbone_bias.get();
        result.first_weights = first_weights.get();
        result.first_biases = first_biases.get();
        result.tail_weights = tail_weights.get();
        result.tail_biases = tail_biases.get();
        result.outputs[0] = out_position.get();
        result.outputs[1] = out_scale.get();
        result.outputs[2] = out_rotation.get();
        result.outputs[3] = out_opacity.get();
        result.outputs[4] = out_sh.get();
        result.rows = deform_tasks * tacker_4dgs::kDeformRows;
        result.persistent_blocks = 1;
        return result;
    }
};

void launch_one(
    RasterBuffers* raster,
    DeformBuffers* deform,
    int raster_tasks,
    int deform_tasks,
    cudaStream_t stream) {
    const CudaRasterizer::MixedDeformMlpTaskV1 task = deform->task(deform_tasks);
    synthetic_mix_render_deform_mlp_v5<<<
        1, CudaRasterizer::Tacker::kMixedThreads, 0, stream>>>(
        raster->ranges.get(), raster->point_list.get(),
        kMaxRasterTasks * BLOCK_X, BLOCK_Y, raster->points.get(),
        raster->features.get(), raster->depths.get(), raster->conics.get(),
        raster->final_t.get(), raster->contributors.get(),
        raster->background.get(), raster->out_color.get(),
        raster->out_depth.get(), raster_tasks, raster_tasks > 0 ? 1 : 0,
        task, 1);
}

double percentile(const std::vector<double>& sorted, double q) {
    const double position = q * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

struct TimingResult {
    double median_us;
    double p10_us;
    double p90_us;
};

TimingResult run_case(
    RasterBuffers* raster,
    DeformBuffers* deform,
    int raster_tasks,
    int deform_tasks,
    int graph_warmup,
    int rounds,
    int launches_per_round) {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int launch = 0; launch < launches_per_round; ++launch) {
        launch_one(raster, deform, raster_tasks, deform_tasks, stream);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    for (int iteration = 0; iteration < graph_warmup; ++iteration) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    std::vector<double> averages;
    averages.reserve(rounds);
    for (int round = 0; round < rounds; ++round) {
        CHECK_CUDA(cudaEventRecord(begin, stream));
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaEventRecord(end, stream));
        CHECK_CUDA(cudaEventSynchronize(end));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
        averages.push_back(
            static_cast<double>(elapsed_ms) * 1000.0 / launches_per_round);
    }
    std::sort(averages.begin(), averages.end());

    CHECK_CUDA(cudaEventDestroy(begin));
    CHECK_CUDA(cudaEventDestroy(end));
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return TimingResult{
        percentile(averages, 0.50),
        percentile(averages, 0.10),
        percentile(averages, 0.90)};
}

struct RasterCase {
    int first_tile_gaussians;
    int second_tile_gaussians;
};

std::vector<int> parse_k_list(const std::string& value) {
    std::vector<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const int parsed = std::stoi(item);
        if (parsed < 0) {
            throw std::runtime_error("Gaussian count cannot be negative");
        }
        result.push_back(parsed);
    }
    if (result.empty()) {
        throw std::runtime_error("Gaussian count list cannot be empty");
    }
    return result;
}

std::vector<RasterCase> symmetric_cases(const std::vector<int>& values) {
    std::vector<RasterCase> result;
    result.reserve(values.size());
    for (int value : values) {
        result.push_back(RasterCase{value, value});
    }
    return result;
}

std::vector<RasterCase> parse_pair_list(const std::string& value) {
    std::vector<RasterCase> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t separator = item.find(':');
        if (separator == std::string::npos ||
            item.find(':', separator + 1) != std::string::npos) {
            throw std::runtime_error(
                "each pair must use the K0:K1 format");
        }
        const int first = std::stoi(item.substr(0, separator));
        const int second = std::stoi(item.substr(separator + 1));
        if (first < 0 || second < 0) {
            throw std::runtime_error("Gaussian counts cannot be negative");
        }
        result.push_back(RasterCase{first, second});
    }
    if (result.empty()) {
        throw std::runtime_error("Raster pair list cannot be empty");
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int graph_warmup = 3;
        int rounds = 11;
        int launches_per_round = 128;
        std::vector<RasterCase> raster_cases = symmetric_cases(
            {0, 128, 256, 384, 512, 640, 768, 1024, 1280, 1536, 1664});
        bool custom_k_list = false;
        bool custom_pair_list = false;
        std::string output_path;
        for (int arg = 1; arg < argc; ++arg) {
            const std::string option = argv[arg];
            if (option == "--graph-warmup" && arg + 1 < argc) {
                graph_warmup = std::stoi(argv[++arg]);
            } else if (option == "--rounds" && arg + 1 < argc) {
                rounds = std::stoi(argv[++arg]);
            } else if (option == "--launches-per-round" && arg + 1 < argc) {
                launches_per_round = std::stoi(argv[++arg]);
            } else if (option == "--k-list" && arg + 1 < argc) {
                if (custom_pair_list) {
                    throw std::runtime_error(
                        "--k-list and --pair-list are mutually exclusive");
                }
                custom_k_list = true;
                raster_cases = symmetric_cases(parse_k_list(argv[++arg]));
            } else if (option == "--pair-list" && arg + 1 < argc) {
                if (custom_k_list) {
                    throw std::runtime_error(
                        "--k-list and --pair-list are mutually exclusive");
                }
                custom_pair_list = true;
                raster_cases = parse_pair_list(argv[++arg]);
            } else if (option == "--output" && arg + 1 < argc) {
                output_path = argv[++arg];
            } else {
                throw std::runtime_error("unknown or incomplete option: " + option);
            }
        }
        if (graph_warmup < 1 || rounds < 1 || launches_per_round < 1) {
            throw std::runtime_error("all repetition counts must be positive");
        }

        int device = 0;
        cudaDeviceProp properties{};
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

        std::ofstream output_file;
        if (!output_path.empty()) {
            output_file.open(output_path);
            if (!output_file) {
                throw std::runtime_error("cannot open output: " + output_path);
            }
        }
        std::ostream& output = output_path.empty() ? std::cout : output_file;
        output << "# device," << properties.name << '\n';
        output << "# compute_capability," << properties.major << '.'
               << properties.minor << '\n';
        output << "# sm_count," << properties.multiProcessorCount << '\n';
        output << "# clock_rate_khz," << properties.clockRate << '\n';
        output << "# graph_warmup," << graph_warmup << '\n';
        output << "# rounds," << rounds << '\n';
        output << "# launches_per_round," << launches_per_round << '\n';
        output << "raster_k0,raster_k1,raster_tasks,deform_tasks,"
                  "median_raw_us,p10_raw_us,p90_raw_us\n";
        output << std::fixed << std::setprecision(4);

        DeformBuffers deform_buffers;
        int maximum_tile_gaussians = 0;
        for (const RasterCase& raster_case : raster_cases) {
            maximum_tile_gaussians = std::max(
                maximum_tile_gaussians,
                std::max(
                    raster_case.first_tile_gaussians,
                    raster_case.second_tile_gaussians));
        }
        RasterBuffers global_warmup_buffers(
            maximum_tile_gaussians, maximum_tile_gaussians);
        (void)run_case(
            &global_warmup_buffers, &deform_buffers, 2, kMaxDeformTasks,
            40, 2, launches_per_round);

        for (const RasterCase& raster_case : raster_cases) {
            RasterBuffers raster_buffers(
                raster_case.first_tile_gaussians,
                raster_case.second_tile_gaussians);
            for (int raster_tasks = 0; raster_tasks <= 2; ++raster_tasks) {
                for (int deform_tasks = 0;
                     deform_tasks <= kMaxDeformTasks;
                     ++deform_tasks) {
                    const TimingResult timing = run_case(
                        &raster_buffers, &deform_buffers,
                        raster_tasks, deform_tasks, graph_warmup, rounds,
                        launches_per_round);
                    output << raster_case.first_tile_gaussians << ','
                           << raster_case.second_tile_gaussians << ','
                           << raster_tasks << ',' << deform_tasks << ','
                           << timing.median_us << ','
                           << timing.p10_us << ',' << timing.p90_us << '\n';
                    output.flush();
                }
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
