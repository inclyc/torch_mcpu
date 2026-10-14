#include <runtime/McpuKernelTiming.h>

#include <algorithm>
#include <array>

#include <c10/util/ApproximateClock.h>

namespace at::mcpu::kernel_timing {
namespace {

std::array<Event, kMaxEventsPerThread> g_events{};
std::atomic<std::size_t> g_event_count{0};

} // namespace

std::atomic<bool> g_enabled{false};

void set_enabled(bool value) {
  g_enabled.store(value, std::memory_order_relaxed);
}

void reset() {
  // The caller must stop submissions and drain workers before resetting slots.
  g_event_count.store(0, std::memory_order_relaxed);
}

Event* reserve_event_slot(const char* name, std::uint64_t stream) {
  if (!enabled()) {
    return nullptr;
  }
  auto index = g_event_count.load(std::memory_order_relaxed);
  do {
    if (C10_UNLIKELY(index >= kMaxEventsPerThread))
      return nullptr;
  } while (!g_event_count.compare_exchange_weak(
      index, index + 1, std::memory_order_relaxed));
  Event& event = g_events[index];
  event = {name, stream, 0, 0};
  return &event;
}

std::size_t event_count() {
  return g_event_count.load(std::memory_order_relaxed);
}

double elapsed_us_between(std::size_t begin, std::size_t end) {
  const std::size_t count = event_count();
  begin = std::min(begin, count);
  end = std::min(end, count);
  if (end <= begin) {
    return 0.0;
  }

  std::uint64_t elapsed_time = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const Event& event = g_events[i];
    if (event.end_time > event.begin_time) {
      elapsed_time += event.end_time - event.begin_time;
    }
  }
  return static_cast<double>(elapsed_time) / 1000.0;
}

std::vector<ThreadSnapshot> snapshot() {
  std::vector<ThreadSnapshot> out;
  out.reserve(1);
  // As with elapsed_us_between, read event payloads only after submission and
  // stream completion. The atomic count makes reservation safe, not live reads.
  const std::size_t count = event_count();
  ThreadSnapshot snapshot;
  snapshot.role = kWorkerRole;
  snapshot.events.assign(g_events.begin(), g_events.begin() + count);
  out.push_back(std::move(snapshot));
  return out;
}

} // namespace at::mcpu::kernel_timing
