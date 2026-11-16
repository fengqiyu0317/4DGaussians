// Opt-in, driver-only adapter; build separately from the qualified CUDA ABI.
// g++ -std=c++17 -O2 -shared -fPIC -I/usr/local/cuda/include green_contexts.cpp
//     -L/usr/lib/aarch64-linux-gnu/nvidia -lcuda -o libtacker_green_contexts.so
#include <cuda.h>
#include <array>
#include <cstdint>
#include <memory>
#include <string>

#if CUDA_VERSION < 12060
#error "Green Contexts adapter requires CUDA 12.6 or newer headers"
#endif

namespace {
struct Pair {
    CUdevice device = 0;
    bool retained = false;
    std::array<CUgreenCtx, 2> green{};
    std::array<CUcontext, 2> converted{};
    std::array<CUstream, 2> streams{};
    std::array<unsigned int, 2> sms{};
};
thread_local std::string last_stage;
}

extern "C" const char* tacker_green_last_stage() { return last_stage.c_str(); }
extern "C" const char* tacker_green_error_name(int code) {
    const char* name = nullptr;
    cuGetErrorName(static_cast<CUresult>(code), &name);
    return name ? name : "unknown";
}

extern "C" int tacker_green_destroy(void* opaque) {
    if (!opaque) return int(CUDA_SUCCESS);
    std::unique_ptr<Pair> pair(static_cast<Pair*>(opaque));
    CUresult first = CUDA_SUCCESS;
    auto record = [&](CUresult code) {
        if (first == CUDA_SUCCESS && code != CUDA_SUCCESS) first = code;
    };
    for (auto stream : pair->streams) if (stream) record(cuStreamSynchronize(stream));
    for (auto stream : pair->streams) if (stream) record(cuStreamDestroy(stream));
    for (auto green : pair->green) if (green) record(cuGreenCtxDestroy(green));
    if (pair->retained) record(cuDevicePrimaryCtxRelease(pair->device));
    return int(first);
}

extern "C" int tacker_green_create(int device_index, int deform_sms, void** output) {
    if (!output) return int(CUDA_ERROR_INVALID_VALUE);
    *output = nullptr;
    if (deform_sms != 4 && deform_sms != 8 && deform_sms != 12) return int(CUDA_ERROR_INVALID_VALUE);
    std::unique_ptr<Pair> pair(new Pair);
    CUcontext previous = nullptr;
    bool have_previous = false;
    auto check = [&](const char* stage, CUresult result) {
        last_stage = stage;
        if (result != CUDA_SUCCESS) throw result;
    };
    try {
        check("cuInit", cuInit(0));
        check("cuDeviceGet", cuDeviceGet(&pair->device, device_index));
        check("cuCtxGetCurrent", cuCtxGetCurrent(&previous));
        have_previous = true;
        CUcontext primary = nullptr;
        check("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&primary, pair->device));
        pair->retained = true;
        check("cuCtxSetCurrent.primary", cuCtxSetCurrent(primary));
        CUdevResource resource{}, remaining{};
        check("cuDeviceGetDevResource", cuDeviceGetDevResource(pair->device, &resource, CU_DEV_RESOURCE_TYPE_SM));
        if (resource.sm.smCount != 16) {
            last_stage = "requires a 16-SM device";
            throw CUDA_ERROR_NOT_SUPPORTED;
        }
        std::array<CUdevResource, 4> groups{};
        unsigned int count = 4;
        check("cuDevSmResourceSplitByCount", cuDevSmResourceSplitByCount(groups.data(), &count, &resource, &remaining, 0, 4));
        if (count != 4) throw CUDA_ERROR_NOT_SUPPORTED;
        const unsigned int deform_groups = unsigned(deform_sms / 4);
        for (unsigned int i = 0; i < 2; ++i) {
            CUdevResourceDesc desc = nullptr;
            check("cuDevResourceGenerateDesc", cuDevResourceGenerateDesc(&desc,
                groups.data() + (i == 0 ? 0 : deform_groups),
                i == 0 ? deform_groups : 4 - deform_groups));
            check("cuGreenCtxCreate", cuGreenCtxCreate(&pair->green[i], desc, pair->device, CU_GREEN_CTX_DEFAULT_STREAM));
            // Required before APIs such as Graph capture access the associated CUcontext.
            check("cuCtxFromGreenCtx", cuCtxFromGreenCtx(&pair->converted[i], pair->green[i]));
            CUdevResource actual{};
            check("cuGreenCtxGetDevResource", cuGreenCtxGetDevResource(pair->green[i], &actual, CU_DEV_RESOURCE_TYPE_SM));
            pair->sms[i] = actual.sm.smCount;
            if (pair->sms[i] != unsigned(i == 0 ? deform_sms : 16 - deform_sms)) {
                last_stage = "created SM count mismatch";
                throw CUDA_ERROR_NOT_SUPPORTED;
            }
            check("cuGreenCtxStreamCreate", cuGreenCtxStreamCreate(&pair->streams[i], pair->green[i], CU_STREAM_NON_BLOCKING, 0));
        }
        check("cuCtxSetCurrent.restore", cuCtxSetCurrent(previous));
        *output = pair.release();
        return int(CUDA_SUCCESS);
    } catch (CUresult result) {
        // Always release the resources owned by a partially created adapter.
        if (have_previous) cuCtxSetCurrent(previous);
        tacker_green_destroy(pair.release());
        return int(result);
    }
}

extern "C" uintptr_t tacker_green_stream(void* opaque, int index) {
    if (!opaque || index < 0 || index > 1) return 0;
    return reinterpret_cast<uintptr_t>(static_cast<Pair*>(opaque)->streams[index]);
}

extern "C" int tacker_green_stream_sms(void* opaque, int index, unsigned int* output) {
    if (!opaque || index < 0 || index > 1 || !output) return int(CUDA_ERROR_INVALID_VALUE);
    auto pair = static_cast<Pair*>(opaque);
    CUgreenCtx attached = nullptr;
    CUresult result = cuStreamGetGreenCtx(pair->streams[index], &attached);
    if (result != CUDA_SUCCESS) return int(result);
    if (attached != pair->green[index]) return int(CUDA_ERROR_INVALID_CONTEXT);
    CUdevResource actual{};
    result = cuGreenCtxGetDevResource(attached, &actual, CU_DEV_RESOURCE_TYPE_SM);
    if (result == CUDA_SUCCESS) *output = actual.sm.smCount;
    return int(result);
}
