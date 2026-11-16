// Driver-only capability probe. Creates/destroys only its own contexts/streams.
// Build with the target's CUDA headers and real libcuda, never the stub library.
#include <cuda.h>
#include <array>
#include <iostream>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

// Called from a PyTorch-initialized process. Keep the opaque CUDA resource ABI
// in C++, using the target's headers instead of guessing its ctypes layout.
extern "C" int probe_sm_resource(unsigned int* count) {
    if (!count) return int(CUDA_ERROR_INVALID_VALUE);
    *count = 0;
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) return int(result);
    CUdevice device = 0;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) return int(result);
    CUdevResource resource{};
    result = cuDeviceGetDevResource(device, &resource, CU_DEV_RESOURCE_TYPE_SM);
    if (result == CUDA_SUCCESS) *count = resource.sm.smCount;
    return int(result);
}

struct GreenPair {
    CUdevice device = 0;
    CUcontext previous = nullptr, primary = nullptr;
    bool retained = false;
    std::array<CUgreenCtx, 2> green{};
    std::array<CUcontext, 2> converted{};
    std::array<CUstream, 2> streams{};
    std::array<unsigned int, 2> sms{};
};

static std::string pair_stage;

extern "C" const char* probe_green_last_stage() { return pair_stage.c_str(); }

extern "C" int probe_green_pair_destroy(void* opaque) {
    if (!opaque) return int(CUDA_SUCCESS);
    std::unique_ptr<GreenPair> pair(static_cast<GreenPair*>(opaque));
    CUresult first_error = CUDA_SUCCESS;
    auto record = [&](CUresult result) {
        if (first_error == CUDA_SUCCESS && result != CUDA_SUCCESS) first_error = result;
    };
    for (auto stream : pair->streams) if (stream) record(cuStreamSynchronize(stream));
    for (auto stream : pair->streams) if (stream) record(cuStreamDestroy(stream));
    for (auto green : pair->green) if (green) record(cuGreenCtxDestroy(green));
    if (pair->retained) {
        record(cuCtxSetCurrent(pair->previous));
        record(cuDevicePrimaryCtxRelease(pair->device));
    }
    return int(first_error);
}

extern "C" int probe_green_pair_create(void** output) {
    if (!output) return int(CUDA_ERROR_INVALID_VALUE);
    *output = nullptr;
    std::unique_ptr<GreenPair> pair(new GreenPair);
    auto check = [&](const char* stage, CUresult result) {
        pair_stage = stage;
        if (result != CUDA_SUCCESS) throw result;
    };
    try {
        check("cuInit", cuInit(0));
        check("cuDeviceGet", cuDeviceGet(&pair->device, 0));
        check("cuCtxGetCurrent", cuCtxGetCurrent(&pair->previous));
        check("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&pair->primary, pair->device));
        pair->retained = true;
        check("cuCtxSetCurrent.primary", cuCtxSetCurrent(pair->primary));
        CUdevResource resource{}, remaining{};
        check("cuDeviceGetDevResource", cuDeviceGetDevResource(pair->device, &resource, CU_DEV_RESOURCE_TYPE_SM));
        if (resource.sm.smCount != 16) throw CUDA_ERROR_NOT_SUPPORTED;
        std::array<CUdevResource, 4> groups{};
        unsigned int count = 4;
        check("cuDevSmResourceSplitByCount", cuDevSmResourceSplitByCount(groups.data(), &count, &resource, &remaining, 0, 4));
        if (count != 4) throw CUDA_ERROR_NOT_SUPPORTED;
        for (unsigned int i = 0; i < 2; ++i) {
            CUdevResourceDesc desc = nullptr;
            check("cuDevResourceGenerateDesc", cuDevResourceGenerateDesc(&desc, groups.data() + (i == 0 ? 0 : 1), i == 0 ? 1 : 3));
            check("cuGreenCtxCreate", cuGreenCtxCreate(&pair->green[i], desc, pair->device, CU_GREEN_CTX_DEFAULT_STREAM));
            check("cuCtxFromGreenCtx", cuCtxFromGreenCtx(&pair->converted[i], pair->green[i]));
            CUdevResource actual{};
            check("cuGreenCtxGetDevResource", cuGreenCtxGetDevResource(pair->green[i], &actual, CU_DEV_RESOURCE_TYPE_SM));
            pair->sms[i] = actual.sm.smCount;
            check("cuGreenCtxStreamCreate", cuGreenCtxStreamCreate(&pair->streams[i], pair->green[i], CU_STREAM_NON_BLOCKING, 0));
        }
        check("cuCtxSetCurrent.restore", cuCtxSetCurrent(pair->previous));
        *output = pair.release();
        return int(CUDA_SUCCESS);
    } catch (CUresult result) {
        probe_green_pair_destroy(pair.release());
        return int(result);
    }
}

extern "C" uintptr_t probe_green_pair_stream(void* opaque, int index) {
    if (!opaque || index < 0 || index > 1) return 0;
    return reinterpret_cast<uintptr_t>(static_cast<GreenPair*>(opaque)->streams[index]);
}

extern "C" unsigned int probe_green_pair_sm_count(void* opaque, int index) {
    if (!opaque || index < 0 || index > 1) return 0;
    return static_cast<GreenPair*>(opaque)->sms[index];
}

static std::string quote(const std::string& value) {
    std::string result = "\"";
    for (char ch : value) {
        if (ch == '\\' || ch == '"') result += '\\';
        if (ch == '\n') result += "\\n";
        else result += ch;
    }
    return result + '"';
}

int main() {
    CUdevice device = 0;
    CUcontext previous = nullptr, primary = nullptr;
    std::array<CUgreenCtx, 2> green{};
    std::array<CUstream, 2> streams{};
    bool retained = false, supported = false;
    int version = 0, sm_count = 0, major = 0, minor = 0;
    char name[256]{};
    std::ostringstream calls;
    bool first = true;
    std::string stopped_at, unexpected;
    CUresult failure = CUDA_SUCCESS;
    auto record = [&](const std::string& api, CUresult result) {
        const char* error_name = nullptr;
        const char* error_string = nullptr;
        cuGetErrorName(result, &error_name);
        cuGetErrorString(result, &error_string);
        if (!first) calls << ',';
        first = false;
        calls << "{\"api\":" << quote(api) << ",\"code\":" << int(result)
              << ",\"name\":" << quote(error_name ? error_name : "unknown")
              << ",\"message\":" << quote(error_string ? error_string : "unknown") << '}';
    };
    auto check = [&](const std::string& api, CUresult result) {
        record(api, result);
        if (result != CUDA_SUCCESS) {
            stopped_at = api;
            failure = result;
            throw result;
        }
    };
    unsigned int resource_sms = 0, groups = 4;
    std::array<unsigned int, 2> partition_sms{};
    try {
        check("cuInit", cuInit(0));
        check("cuDriverGetVersion", cuDriverGetVersion(&version));
        check("cuDeviceGet", cuDeviceGet(&device, 0));
        check("cuDeviceGetName", cuDeviceGetName(name, sizeof(name), device));
        check("cuDeviceGetAttribute.SM_COUNT", cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
        check("cuDeviceGetAttribute.CC_MAJOR", cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        check("cuDeviceGetAttribute.CC_MINOR", cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        check("cuCtxGetCurrent", cuCtxGetCurrent(&previous));
        check("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&primary, device));
        retained = true;
        check("cuCtxSetCurrent.primary", cuCtxSetCurrent(primary));
        CUdevResource resource{};
        check("cuDeviceGetDevResource.SM", cuDeviceGetDevResource(device, &resource, CU_DEV_RESOURCE_TYPE_SM));
        resource_sms = resource.sm.smCount;
        if (resource_sms != 16) throw std::runtime_error("This probe's 4+12 layout requires 16 SMs");
        std::array<CUdevResource, 4> partitions{};
        CUdevResource remaining{};
        check("cuDevSmResourceSplitByCount.4x4", cuDevSmResourceSplitByCount(
            partitions.data(), &groups, &resource, &remaining, 0, 4));
        if (groups != 4) throw std::runtime_error("Driver did not produce four 4-SM groups");
        for (unsigned int i = 0; i < 2; ++i) {
            CUdevResourceDesc desc = nullptr;
            const auto suffix = std::string(i == 0 ? ".4sm" : ".12sm");
            check("cuDevResourceGenerateDesc" + suffix, cuDevResourceGenerateDesc(
                &desc, partitions.data() + (i == 0 ? 0 : 1), i == 0 ? 1 : 3));
            check("cuGreenCtxCreate" + suffix, cuGreenCtxCreate(&green[i], desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
            CUcontext converted = nullptr;
            check("cuCtxFromGreenCtx" + suffix, cuCtxFromGreenCtx(&converted, green[i]));
            CUdevResource actual{};
            check("cuGreenCtxGetDevResource" + suffix, cuGreenCtxGetDevResource(green[i], &actual, CU_DEV_RESOURCE_TYPE_SM));
            partition_sms[i] = actual.sm.smCount;
            check("cuGreenCtxStreamCreate" + suffix, cuGreenCtxStreamCreate(&streams[i], green[i], CU_STREAM_NON_BLOCKING, 0));
            check("cuStreamSynchronize" + suffix, cuStreamSynchronize(streams[i]));
        }
        supported = partition_sms[0] == 4 && partition_sms[1] == 12;
        if (!supported) throw std::runtime_error("Created SM counts did not match 4+12");
    } catch (CUresult) {
        // Preserve the exact API and driver result, including NOT_SUPPORTED.
    } catch (const std::exception& error) {
        unexpected = error.what();
    }
    for (auto stream : streams) if (stream) record("cleanup.cuStreamDestroy", cuStreamDestroy(stream));
    for (auto context : green) if (context) record("cleanup.cuGreenCtxDestroy", cuGreenCtxDestroy(context));
    if (retained) {
        record("cleanup.cuCtxSetCurrent", cuCtxSetCurrent(previous));
        record("cleanup.cuDevicePrimaryCtxRelease", cuDevicePrimaryCtxRelease(device));
    }
    std::cout << "{\"header_cuda_version\":" << CUDA_VERSION
              << ",\"driver_cuda_version\":" << version
              << ",\"device\":" << quote(name) << ",\"sm_count\":" << sm_count
              << ",\"compute_capability\":[" << major << ',' << minor << ']'
              << ",\"resource_sm_count\":" << resource_sms
              << ",\"created_partition_sm_counts\":[" << partition_sms[0] << ',' << partition_sms[1] << ']'
              << ",\"supported\":" << (supported ? "true" : "false")
              << ",\"stopped_at\":" << quote(stopped_at)
              << ",\"failure_code\":" << int(failure)
              << ",\"unexpected_error\":" << quote(unexpected)
              << ",\"calls\":[" << calls.str() << "]}\n";
    return unexpected.empty() && (failure == CUDA_SUCCESS || failure == CUDA_ERROR_NOT_SUPPORTED) ? 0 : 2;
}
