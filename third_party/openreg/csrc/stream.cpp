#include <include/openreg.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

static std::mutex g_mutex;
static std::once_flag g_flag;
static std::vector<std::set<orStream_t>> g_streams_per_device;
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
thread_local bool g_in_kernel_task = false;
#endif

#ifndef TORCH_MCPU_STREAM_IDLE_BUSY
#define TORCH_MCPU_STREAM_IDLE_BUSY 0
#endif
#ifndef TORCH_MCPU_STREAM_IDLE_HYBRID
#define TORCH_MCPU_STREAM_IDLE_HYBRID 1
#endif
#ifndef TORCH_MCPU_STREAM_IDLE_BLOCK
#define TORCH_MCPU_STREAM_IDLE_BLOCK 2
#endif
#ifndef TORCH_MCPU_HIGH_PRIORITY_STREAM_IDLE_POLICY
#define TORCH_MCPU_HIGH_PRIORITY_STREAM_IDLE_POLICY TORCH_MCPU_STREAM_IDLE_BUSY
#endif
#ifndef TORCH_MCPU_LOW_PRIORITY_STREAM_IDLE_POLICY
#define TORCH_MCPU_LOW_PRIORITY_STREAM_IDLE_POLICY TORCH_MCPU_STREAM_IDLE_BLOCK
#endif
#ifndef TORCH_MCPU_STREAM_IDLE_SPIN_ITERS
#define TORCH_MCPU_STREAM_IDLE_SPIN_ITERS 5000
#endif

#define TORCH_MCPU_STREAM_IDLE_POLICY_VALID(policy) \
  ((policy) == TORCH_MCPU_STREAM_IDLE_BUSY ||       \
   (policy) == TORCH_MCPU_STREAM_IDLE_HYBRID ||     \
   (policy) == TORCH_MCPU_STREAM_IDLE_BLOCK)

static_assert(
    TORCH_MCPU_STREAM_IDLE_POLICY_VALID(
        TORCH_MCPU_HIGH_PRIORITY_STREAM_IDLE_POLICY),
    "TORCH_MCPU_HIGH_PRIORITY_STREAM_IDLE_POLICY must be "
    "TORCH_MCPU_STREAM_IDLE_BUSY, TORCH_MCPU_STREAM_IDLE_HYBRID, or "
    "TORCH_MCPU_STREAM_IDLE_BLOCK");
static_assert(
    TORCH_MCPU_STREAM_IDLE_POLICY_VALID(
        TORCH_MCPU_LOW_PRIORITY_STREAM_IDLE_POLICY),
    "TORCH_MCPU_LOW_PRIORITY_STREAM_IDLE_POLICY must be "
    "TORCH_MCPU_STREAM_IDLE_BUSY, TORCH_MCPU_STREAM_IDLE_HYBRID, or "
    "TORCH_MCPU_STREAM_IDLE_BLOCK");
static_assert(
    TORCH_MCPU_STREAM_IDLE_SPIN_ITERS >= 0,
    "TORCH_MCPU_STREAM_IDLE_SPIN_ITERS must be non-negative");

static void initialize_registries() {
  int device_count = 0;
  orGetDeviceCount(&device_count);
  g_streams_per_device.resize(device_count);
}

static bool parse_non_negative_int(const char* value, int* out) {
  if (!value || *value == '\0' || !out) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

static int parse_worker_core(const char* value, const char* env_name) {
  if (!value || *value == '\0' || std::strcmp(value, "none") == 0) {
    return -1;
  }

  int core = -1;
  if (!parse_non_negative_int(value, &core)) {
    std::cerr << "Invalid " << env_name << "='" << value
              << "'; default stream worker affinity disabled\n";
    return -1;
  }
  return core;
}

static int choose_worker_core(bool is_default_stream) {
  if (!is_default_stream) {
    return -1;
  }
  const char* legacy = std::getenv("TORCH_MCPU_STREAM_WORKER_CORE");
  if (!legacy) {
    // Do not pin production stream workers by default. OpenMP kernels launched
    // from the worker inherit the worker affinity mask, so single-core pinning
    // collapses OMP_NUM_THREADS parallel regions onto one CPU.
    return -1;
  }

  return parse_worker_core(legacy, "TORCH_MCPU_STREAM_WORKER_CORE");
}

static orStream::IdlePolicy idle_policy_from_macro(int policy) {
  switch (policy) {
    case TORCH_MCPU_STREAM_IDLE_BUSY:
      return orStream::IdlePolicy::Busy;
    case TORCH_MCPU_STREAM_IDLE_HYBRID:
      return orStream::IdlePolicy::Hybrid;
    case TORCH_MCPU_STREAM_IDLE_BLOCK:
      return orStream::IdlePolicy::Block;
  }
  return orStream::IdlePolicy::Busy;
}

static bool is_highest_priority(int priority) {
  int least_priority = 0;
  int greatest_priority = 0;
  if (orDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) !=
      orSuccess) {
    return false;
  }
  return priority == greatest_priority;
}

static orStream::IdlePolicy choose_idle_policy(
    int priority,
    bool is_default_stream) {
  if (is_default_stream) {
    return orStream::IdlePolicy::Busy;
  }

  const bool high_priority = is_highest_priority(priority);
  return idle_policy_from_macro(high_priority
      ? TORCH_MCPU_HIGH_PRIORITY_STREAM_IDLE_POLICY
      : TORCH_MCPU_LOW_PRIORITY_STREAM_IDLE_POLICY);
}

static std::uint64_t choose_spin_iters() {
  return static_cast<std::uint64_t>(TORCH_MCPU_STREAM_IDLE_SPIN_ITERS);
}

static void pin_this_thread_to_core(int core) {
#if defined(__linux__)
  if (core < 0) {
    return;
  }
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  int status =
      pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  if (status != 0) {
    std::cerr << "Failed to set stream worker affinity: "
              << std::strerror(status) << "\n";
  }
#else
  (void)core;
#endif
}

struct orEventRecordState {
  std::mutex mtx;
  std::condition_variable cv;
  bool completed = false;
  std::uint64_t version = 0;
  std::chrono::high_resolution_clock::time_point completion_time;
};

struct orEventImpl {
  std::mutex mtx;
  std::uint64_t next_version = 0;
  std::shared_ptr<orEventRecordState> latest_record;
  int device_index = -1;
  bool timing_enabled{false};
};

struct orEvent {
  std::shared_ptr<orEventImpl> impl;
};

orStream::orStream() : orStream(0, false) {}

orStream::orStream(int stream_priority, bool default_stream)
    : priority(stream_priority),
      is_default_stream(default_stream),
      idle_policy(choose_idle_policy(stream_priority, default_stream)),
      spin_iters(choose_spin_iters()),
      affinity_core(choose_worker_core(default_stream)) {
  for (std::uint64_t i = 0; i < kQueueCapacity; ++i) {
    slots[i].sequence.store(i, std::memory_order_relaxed);
  }
  worker = std::thread([this] {
    pin_this_thread_to_core(affinity_core);
    workerLoop();
  });
}

orStream::~orStream() {
  synchronize();
  stop.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(wake_mutex);
    wake_cv.notify_all();
  }
  if (worker.joinable()) {
    worker.join();
  }
}

template <orStream::IdlePolicy Policy>
static void workerLoopImpl(orStream* stream) {
  std::uint64_t local_head = stream->head.load(std::memory_order_relaxed);
  std::uint64_t idle_spins = 0;
  while (true) {
    auto& slot = stream->slots[local_head & orStream::kQueueMask];
    const auto sequence = slot.sequence.load(std::memory_order_acquire);
    if (sequence != local_head + 1) {
      if (stream->stop.load(std::memory_order_acquire) &&
          local_head == stream->tail.load(std::memory_order_acquire)) {
        return;
      }

      if constexpr (Policy == orStream::IdlePolicy::Busy) {
        openreg::cpu_relax();
        continue;
      }

      if constexpr (Policy == orStream::IdlePolicy::Hybrid) {
        if (idle_spins++ < stream->spin_iters) {
          openreg::cpu_relax();
          continue;
        }
      }

      std::unique_lock<std::mutex> lock(stream->wake_mutex);
      stream->wake_cv.wait(lock, [&] {
        return slot.sequence.load(std::memory_order_acquire) ==
            local_head + 1 ||
            stream->stop.load(std::memory_order_acquire);
      });
      idle_spins = 0;
      continue;
    }

    slot.run(slot.storage);
    slot.destroy(slot.storage);
    slot.sequence.store(
        local_head + orStream::kQueueCapacity, std::memory_order_release);
    ++local_head;
    stream->head.store(local_head, std::memory_order_release);
    stream->completed.store(local_head, std::memory_order_release);
    idle_spins = 0;
  }
}

void orStream::workerLoop() {
  switch (idle_policy) {
    case IdlePolicy::Busy:
      workerLoopImpl<IdlePolicy::Busy>(this);
      return;
    case IdlePolicy::Hybrid:
      workerLoopImpl<IdlePolicy::Hybrid>(this);
      return;
    case IdlePolicy::Block:
      workerLoopImpl<IdlePolicy::Block>(this);
      return;
  }
}

void orStream::synchronize() {
  const auto target = tail.load(std::memory_order_acquire);
  while (completed.load(std::memory_order_acquire) < target) {
    openreg::cpu_relax();
  }
}

bool orStream::idle() const {
  return completed.load(std::memory_order_acquire) ==
      tail.load(std::memory_order_acquire);
}

void orStream::notifyWorker() {
  if (idle_policy == IdlePolicy::Busy) {
    return;
  }
  // Pair publication with the worker's predicate-check/wait transition. Without
  // this lock, notification can be lost between that check and entering wait.
  std::lock_guard<std::mutex> lock(wake_mutex);
  wake_cv.notify_one();
}

#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
bool openreg::isInKernelTask() {
  return g_in_kernel_task;
}

void openreg::setInKernelTask(bool enabled) {
  g_in_kernel_task = enabled;
}
#endif

orError_t orEventCreateWithFlags(orEvent_t* event, unsigned int flags) {
  if (!event)
    return orErrorUnknown;

  auto impl = std::make_shared<orEventImpl>();
  orGetDevice(&(impl->device_index));
  if (flags & orEventEnableTiming) {
    impl->timing_enabled = true;
  }

  *event = new orEvent{std::move(impl)};
  return orSuccess;
}

orError_t orEventCreate(orEvent_t* event) {
  return orEventCreateWithFlags(event, orEventDisableTiming);
}

orError_t orEventDestroy(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  delete event;
  return orSuccess;
}

orError_t orEventRecord(orEvent_t event, orStream_t stream) {
  if (!event || !stream)
    return orErrorUnknown;

  auto event_impl = event->impl;
  if (event_impl->device_index != stream->device_index) {
    return orErrorUnknown;
  }

  auto record = std::make_shared<orEventRecordState>();
  const bool timing_enabled = event_impl->timing_enabled;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record->version = ++event_impl->next_version;
    event_impl->latest_record = record;
  }

  return orLaunchKernel(stream, [record, timing_enabled]() {
    std::lock_guard<std::mutex> lock(record->mtx);
    if (timing_enabled) {
      record->completion_time = std::chrono::high_resolution_clock::now();
    }
    record->completed = true;
    record->cv.notify_all();
  });
}

orError_t orEventSynchronize(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  std::shared_ptr<orEventRecordState> record;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record = event_impl->latest_record;
  }

  if (!record) {
    return orSuccess;
  }

  std::unique_lock<std::mutex> lock(record->mtx);
  record->cv.wait(lock, [&] {
    return record->completed;
  });

  return orSuccess;
}

orError_t orEventQuery(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  std::shared_ptr<orEventRecordState> record;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record = event_impl->latest_record;
  }

  if (!record) {
    return orSuccess;
  }

  std::lock_guard<std::mutex> lock(record->mtx);
  return record->completed ? orSuccess : orErrorNotReady;
}

orError_t orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end) {
  if (!ms || !start || !end)
    return orErrorUnknown;

  auto start_impl = start->impl;
  auto end_impl = end->impl;

  if (start_impl->device_index != end_impl->device_index) {
    return orErrorUnknown;
  }

  if (!start_impl->timing_enabled || !end_impl->timing_enabled) {
    return orErrorUnknown;
  }

  auto snapshot_record = [](const std::shared_ptr<orEventImpl>& impl) {
    std::lock_guard<std::mutex> lock(impl->mtx);
    return impl->latest_record;
  };

  auto start_record = snapshot_record(start_impl);
  auto end_record = snapshot_record(end_impl);
  if (!start_record || !end_record) {
    return orErrorUnknown;
  }

  auto snapshot_completion_time =
      [](const std::shared_ptr<orEventRecordState>& record,
         auto* completion_time) {
        std::lock_guard<std::mutex> lock(record->mtx);
        if (!record->completed) {
          return false;
        }
        *completion_time = record->completion_time;
        return true;
      };

  std::chrono::high_resolution_clock::time_point start_completion_time;
  std::chrono::high_resolution_clock::time_point end_completion_time;
  if (!snapshot_completion_time(start_record, &start_completion_time) ||
      !snapshot_completion_time(end_record, &end_completion_time)) {
    return orErrorUnknown;
  }

  auto duration = end_completion_time - start_completion_time;
  *ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            duration)
            .count();

  return orSuccess;
}

orError_t orStreamCreateWithPriority(
    orStream_t* stream,
    unsigned int flag,
    int priority) {
  if (!stream) {
    return orErrorUnknown;
  }

  int least_priority, greatest_priority;
  orDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  const int min_priority = std::min(least_priority, greatest_priority);
  const int max_priority = std::max(least_priority, greatest_priority);
  if (priority < min_priority || priority > max_priority) {
    return orErrorUnknown;
  }

  int current_device = 0;
  orGetDevice(&current_device);

  const bool is_default_stream = (flag & orStreamDefault) != 0;
  orStream_t new_stream = new orStream(priority, is_default_stream);
  new_stream->device_index = current_device;

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::call_once(g_flag, initialize_registries);
    g_streams_per_device[current_device].insert(new_stream);
  }

  *stream = new_stream;

  return orSuccess;
}

orError_t orStreamCreate(orStream_t* stream) {
  int least_priority, greatest_priority;
  orDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

  return orStreamCreateWithPriority(stream, 0, greatest_priority);
}

orError_t orStreamGetPriority(orStream_t stream, int* priority) {
  if (!stream || !priority) {
    return orErrorUnknown;
  }
  *priority = stream->priority;
  return orSuccess;
}

orError_t orStreamDestroy(orStream_t stream) {
  if (!stream)
    return orErrorUnknown;

  {
    std::lock_guard<std::mutex> lock(g_mutex);

    int device_idx = stream->device_index;
    if (device_idx >= 0 &&
        device_idx < static_cast<int>(g_streams_per_device.size())) {
      g_streams_per_device[device_idx].erase(stream);
    }
  }

  delete stream;
  return orSuccess;
}

orError_t orStreamQuery(orStream_t stream) {
  if (!stream) {
    return orErrorUnknown;
  }

  return stream->idle() ? orSuccess : orErrorNotReady;
}

orError_t orStreamSynchronize(orStream_t stream) {
  if (!stream)
    return orErrorUnknown;

  stream->synchronize();
  return orSuccess;
}

orError_t orStreamWaitEvent(orStream_t stream, orEvent_t event, unsigned int) {
  if (!stream || !event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  std::shared_ptr<orEventRecordState> record;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record = event_impl->latest_record;
  }

  if (!record) {
    return orSuccess;
  }

  return orLaunchKernel(stream, [record]() {
    std::unique_lock<std::mutex> lock(record->mtx);
    record->cv.wait(lock, [&] {
      return record->completed;
    });
  });
}

orError_t orDeviceGetStreamPriorityRange(
    int* leastPriority,
    int* greatestPriority) {
  if (!leastPriority || !greatestPriority) {
    return orErrorUnknown;
  }

  *leastPriority = 1;
  *greatestPriority = 0;
  return orSuccess;
}

orError_t orDeviceSynchronize(void) {
  int current_device = 0;
  orGetDevice(&current_device);

  std::vector<orStream_t> streams;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::call_once(g_flag, initialize_registries);

    auto& streams_on_device = g_streams_per_device[current_device];
    streams.assign(streams_on_device.begin(), streams_on_device.end());
  }

  for (orStream_t stream : streams) {
    orError_t status = orStreamSynchronize(stream);
    if (status != orSuccess) {
      return status;
    }
  }

  return orSuccess;
}
