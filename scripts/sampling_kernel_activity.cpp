// GPU concurrent-kernel timestamps only. No runtime/driver/NVTX callbacks,
// no event nodes added to CUDA Graphs, and no I/O in CUPTI callbacks.
// API: https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

struct Record {
  uint64_t start, end, node;
  uint32_t stream, context;
  int32_t grid, block;
  uint16_t registers;
  std::string name;
};
static std::mutex lock;
static std::vector<Record> records;
static std::atomic<size_t> dropped{0};
static std::atomic<int> callback_error{0};
static bool initialized = false;

static void CUPTIAPI requested(uint8_t **buffer, size_t *size,
                              size_t *max_records) {
  *size = 16 * 1024 * 1024;
  *max_records = 0;
  void *memory = nullptr;
  if (posix_memalign(&memory, 8, *size) != 0) {
    callback_error.store(-1);
    *size = 0;
  }
  *buffer = static_cast<uint8_t *>(memory);
}

static void CUPTIAPI completed(CUcontext context, uint32_t stream,
                              uint8_t *buffer, size_t, size_t valid_size) {
  std::lock_guard<std::mutex> guard(lock);
  CUpti_Activity *activity = nullptr;
  CUptiResult status;
  while ((status = cuptiActivityGetNextRecord(buffer, valid_size,
                                             &activity)) == CUPTI_SUCCESS) {
    if (activity->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      auto *kernel = reinterpret_cast<CUpti_ActivityKernel9 *>(activity);
      records.push_back({kernel->start, kernel->end, kernel->graphNodeId,
                         kernel->streamId, kernel->contextId, kernel->gridX, kernel->blockX,
                         kernel->registersPerThread, kernel->name ? kernel->name : ""});
    }
  }
  if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) callback_error.store(status);
  size_t lost = 0;
  status = cuptiActivityGetNumDroppedRecords(context, stream, &lost);
  if (status != CUPTI_SUCCESS) callback_error.store(status);
  dropped.fetch_add(lost);
  free(buffer);
}

extern "C" int sampling_activity_init() {
  if (initialized) return -2;
  records.reserve(100000);
  CUptiResult status = cuptiActivityRegisterCallbacks(requested, completed);
  if (status != CUPTI_SUCCESS) return status;
  // CONCURRENT_KERNEL is essential: the KERNEL activity kind serializes work.
  status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (status == CUPTI_SUCCESS) initialized = true;
  return status;
}

extern "C" int sampling_activity_clear() {
  CUptiResult status = cuptiActivityFlushAll(0);
  if (status != CUPTI_SUCCESS) return status;
  if (callback_error.load() != 0) return callback_error.load();
  std::lock_guard<std::mutex> guard(lock);
  records.clear();
  dropped.store(0);
  return 0;
}

extern "C" int sampling_activity_write(const char *path) {
  // Caller has already completed the normal benchmark's sequence-end sync.
  CUptiResult status = cuptiActivityFlushAll(0);
  if (status != CUPTI_SUCCESS) return status;
  if (callback_error.load() != 0) return callback_error.load();
  if (dropped.load() != 0) return -3;
  std::lock_guard<std::mutex> guard(lock);
  std::ofstream file(path);
  if (!file) return -4;
  file << "start_ns,end_ns,stream_id,graph_node_id,grid_x,block_x,registers,context_id,name\n";
  for (const auto &record : records) {
    if (record.start == 0 || record.end < record.start) return -5;
    file << record.start << ',' << record.end << ',' << record.stream << ','
         << record.node << ',' << record.grid << ',' << record.block << ','
         << record.registers << ',' << record.context << ",\"";
    for (char c : record.name) {
      if (c == '"') file << '"';
      file << c;
    }
    file << "\"\n";
  }
  return file.good() ? 0 : -6;
}

extern "C" int sampling_activity_stop() {
  return cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}
