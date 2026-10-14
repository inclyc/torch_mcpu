#include "Extra.h"

#include "runtime/McpuKernelLaunch.h"
#include "runtime/OpenRegException.h"
#include "runtime/OpenRegStream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace at::native::mcpu {

namespace {

void check_stream_test_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1);
  TORCH_CHECK(
      tensor.scalar_type() == at::kLong,
      "stream test ops only support int64 tensors");
  TORCH_CHECK(
      tensor.is_contiguous(), "stream test ops require contiguous input");
  TORCH_CHECK(tensor.numel() > 0, "stream test ops require non-empty input");
}

void sleep_for_ms(int64_t sleep_ms) {
  TORCH_CHECK(sleep_ms >= 0, "sleep_ms must be non-negative");
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
}

std::atomic<int64_t> g_lifetime_payload_constructed{0};
std::atomic<int64_t> g_lifetime_payload_destroyed{0};
std::atomic<int64_t> g_lifetime_payload_ran{0};

struct KernelLaunchLifetimePayload {
  explicit KernelLaunchLifetimePayload(int64_t value) : value(value) {
    g_lifetime_payload_constructed.fetch_add(1, std::memory_order_relaxed);
  }

  ~KernelLaunchLifetimePayload() {
    g_lifetime_payload_destroyed.fetch_add(1, std::memory_order_relaxed);
  }

  int64_t value;
};

} // namespace

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}

int64_t _fused_sdp_choice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
}

void quantize_tensor_per_tensor_affine_stub(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_v = value.size(3);
  const int64_t max_seqlen_q = query.size(2);
  const int64_t max_seqlen_kv = key.size(2);

  auto opts = query.options();
  auto output =
      at::empty({batch_size, num_heads, max_seqlen_q, head_dim_v}, opts);
  auto logsumexp =
      at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  auto debug_attn_mask = at::empty(
      {batch_size, num_heads, max_seqlen_q, max_seqlen_kv},
      opts.dtype(at::kFloat));
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  return std::make_tuple(
      output,
      logsumexp,
      at::Tensor(),
      at::Tensor(),
      max_seqlen_q,
      max_seqlen_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      at::empty_like(query),
      at::empty_like(key),
      at::empty_like(value),
      at::empty_like(attn_bias));
}

namespace {
struct CustomAutogradFnReturnsSelf
    : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing
    : public torch::autograd::Function<CustomAutogradFnAliasing> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};
} // namespace

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

/*
 This implementation is only used to test stub registration, so not all
 capabilities are fully supported.

 Current Limitations:
 - dtype: Float only
 - input tensor: must be contiguous layout
*/
// LITERALINCLUDE START: STUB ABS
void abs_kernel(at::TensorIteratorBase& iter) {
  TORCH_CHECK(iter.ntensors() == 2, "Abs kernel expects 2 tensors");
  TORCH_CHECK(
      iter.common_dtype() == at::ScalarType::Float,
      "Abs kernel only supports float type");

  auto& output_tensor = iter.tensor(0);
  auto& input_tensor = iter.tensor(1);

  TORCH_CHECK(
      input_tensor.sizes() == output_tensor.sizes(),
      "Input and output tensor sizes must match.");

  auto abs_loop = [](float* out_ptr, const float* in_ptr, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = std::abs(in_ptr[i]);
    }
  };

  MemoryGuard guard(input_tensor, output_tensor);

  if (iter.is_contiguous()) {
    abs_loop(
        static_cast<float*>(iter.data_ptr(0)),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());
  } else {
    TORCH_CHECK(
        input_tensor.is_contiguous(), "Input tensor must be contiguous.")

    auto output = at::empty(
        input_tensor.sizes(),
        input_tensor.options().memory_format(
            input_tensor.suggest_memory_format()));

    MemoryGuard guard(output);

    abs_loop(
        static_cast<float*>(output.data_ptr()),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());

    output_tensor.copy_(output);
  }
}
// LITERALINCLUDE END: STUB ABS

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

at::Tensor custom_abs(at::Tensor x) {
  return at::abs(x);
}

at::Tensor& stream_sleep_fill_(at::Tensor& x, int64_t value, int64_t sleep_ms) {
  check_stream_test_tensor(x);

  at::mcpu::launch_timed_kernel(
      "mcpu::stream_sleep_fill_",
      [x, value, sleep_ms](
          at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::stream_sleep_fill_", timing_event);
        sleep_for_ms(sleep_ms);
        at::mcpu::KernelMemoryGuard guard(x);
        auto* data = x.data_ptr<int64_t>();
        std::fill_n(data, x.numel(), value);
      });

  return x;
}

at::Tensor& stream_sleep_copy_(
    at::Tensor& dst,
    const at::Tensor& src,
    int64_t sleep_ms) {
  check_stream_test_tensor(dst);
  check_stream_test_tensor(src);
  TORCH_CHECK(
      dst.sizes() == src.sizes(),
      "stream_sleep_copy_ expects src/dst with identical shapes");

  at::mcpu::launch_kernel(dst, [dst, src, sleep_ms]() mutable {
    sleep_for_ms(sleep_ms);
    at::mcpu::KernelMemoryGuard guard(dst, src);
    std::copy_n(src.data_ptr<int64_t>(), src.numel(), dst.data_ptr<int64_t>());
  });

  return dst;
}

int64_t first_element_int(const at::Tensor& x) {
  check_stream_test_tensor(x);
  MemoryGuard guard(x);
  return x.data_ptr<int64_t>()[0];
}

// Runtime test probe: mutable stream predicates, skipped payload destruction,
// large move-only closure spilling, and restoration of unconditional
// submission.
std::tuple<int64_t, int64_t, int64_t> kernel_launch_condition_counts(
    const at::Tensor& selector) {
  check_stream_test_tensor(selector);
  g_lifetime_payload_constructed.store(0, std::memory_order_relaxed);
  g_lifetime_payload_destroyed.store(0, std::memory_order_relaxed);
  g_lifetime_payload_ran.store(0, std::memory_order_relaxed);
  auto enabled = std::make_shared<bool>(false);
  std::weak_ptr<bool> lifetime = enabled;
  for (int i = 0; i < 4; ++i) {
    at::mcpu::launch_kernel([enabled, i] { *enabled = i % 2 == 0; });
    at::mcpu::KernelLaunchConditionScope condition(enabled);
    if (i == 0) {
      bool rejected_nested = false, rejected_stream = false;
      try {
        at::mcpu::KernelLaunchConditionScope nested(enabled);
      } catch (const c10::Error&) {
        rejected_nested = true;
      }
      try {
        at::mcpu::launch_kernel_on_stream(
            c10::mcpu::getStreamFromPool(), [] {});
      } catch (const c10::Error&) {
        rejected_stream = true;
      }
      TORCH_CHECK(
          rejected_nested && rejected_stream,
          "unsafe launch condition accepted");
    }
    at::mcpu::launch_kernel(
        [payload = std::make_unique<KernelLaunchLifetimePayload>(i)] {
          TORCH_CHECK(payload != nullptr);
          g_lifetime_payload_ran.fetch_add(1, std::memory_order_relaxed);
        });
    // The ordinary timed closure fits exactly; adding the condition pointer
    // must spill even when the queue's inline capacity changes.
    constexpr size_t large_count =
        (sizeof(openreg::QueueSlot::storage) - 32) / sizeof(int64_t);
    std::array<int64_t, large_count> large{};
    large.back() = i;
    const int64_t extra = i + 1;
    at::mcpu::launch_timed_kernel(
        "mcpu::conditional_large_test",
        [payload = std::make_unique<KernelLaunchLifetimePayload>(i),
         large,
         i,
         extra](at::mcpu::kernel_timing::Event* event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::conditional_large_test", event);
          TORCH_CHECK(
              payload != nullptr && large.back() == i && extra == i + 1);
          g_lifetime_payload_ran.fetch_add(1, std::memory_order_relaxed);
        });
  }
  at::mcpu::launch_kernel(
      [] { g_lifetime_payload_ran.fetch_add(1, std::memory_order_relaxed); });
  enabled.reset();
  TORCH_CHECK(
      orStreamSynchronize(c10::mcpu::getCurrentMcpuStream()) == orSuccess);
  TORCH_CHECK(
      lifetime.expired(),
      "conditional launch retained the predicate after execution");
  return {
      g_lifetime_payload_constructed.load(),
      g_lifetime_payload_destroyed.load(),
      g_lifetime_payload_ran.load()};
}

std::tuple<int64_t, int64_t, int64_t> kernel_launch_condition_lifetime_counts(
    const at::Tensor& selector,
    bool enabled,
    bool fail_submission) {
  check_stream_test_tensor(selector);
  const auto stream = c10::mcpu::getCurrentMcpuStream();
  const auto other_stream = c10::mcpu::getStreamFromPool();
  std::atomic<int64_t> destroyed{0}, ran{0};
  std::atomic<bool> release{false};
  struct State {
    explicit State(std::atomic<int64_t>* count) : destroyed(count) {}
    ~State() {
      destroyed->fetch_add(1);
    }
    std::atomic<int64_t>* destroyed;
    bool enabled = false;
  };
  auto state = std::make_shared<State>(&destroyed);
  std::weak_ptr<State> weak = state;
  auto* ptr = state.get();
  // Even a failed assertion must unblock/drain tasks before their stack data
  // dies.
  struct Drain {
    orStream_t stream;
    std::atomic<bool>& release;
    ~Drain() {
      release.store(true);
      (void)orStreamSynchronize(stream);
    }
  } drain{stream, release};
  at::mcpu::launch_kernel([&release] {
    while (!release.load())
      std::this_thread::yield();
  });
  int64_t max_owners = 0;
  bool caught = false;
  try {
    at::mcpu::KernelLaunchLifetimeScope lifetime(state);
    state.reset();
    // The invocation owner alone protects even the unconditional producer.
    at::mcpu::launch_kernel([ptr, enabled] { ptr->enabled = enabled; });
    {
      at::mcpu::KernelLaunchConditionScope condition(lifetime, &ptr->enabled);
      at::mcpu::KernelLaunchConditionScope empty(nullptr);
      for (int i = 0; i < 32; ++i) {
        at::mcpu::launch_kernel([&ran] { ran.fetch_add(1); });
        max_owners = std::max<int64_t>(max_owners, weak.use_count());
      }
      TORCH_CHECK(
          max_owners == 1, "conditional tasks acquired per-task owners");
      if (fail_submission) {
        struct ThrowOnMove {
          ThrowOnMove() = default;
          ThrowOnMove(ThrowOnMove&&) {
            throw std::runtime_error("test enqueue failure");
          }
          void operator()() const {}
        };
        // Exercise a failed queue placement after a successfully submitted
        // prefix.
        const auto status = orLaunchKernel(stream, ThrowOnMove{});
        TORCH_CHECK(status == orSuccess, "injected enqueue failure");
      }
    }
    // The lifetime must retire on its original stream, not the current one.
    c10::mcpu::setCurrentMcpuStream(other_stream);
  } catch (const c10::Error& error) {
    caught = std::string(error.what()).find("injected enqueue failure") !=
        std::string::npos;
    if (!caught)
      throw;
  }
  c10::mcpu::setCurrentMcpuStream(stream);
  TORCH_CHECK(caught == fail_submission);
  TORCH_CHECK(!weak.expired(), "owner released before blocked tasks completed");
  at::mcpu::launch_kernel([&ran] { ran.fetch_add(1); });
  release.store(true);
  TORCH_CHECK(orStreamSynchronize(stream) == orSuccess);
  TORCH_CHECK(
      weak.expired(), "invocation owner retained after stream completion");
  return {destroyed.load(), ran.load(), max_owners};
}

std::tuple<int64_t, int64_t> kernel_launch_parallel_condition_counts(
    const at::Tensor& selector,
    int64_t launches) {
  check_stream_test_tensor(selector);
  TORCH_CHECK(launches > 0 && launches <= 4096);
  constexpr int count = 4;
  std::vector<c10::mcpu::McpuStream> streams;
  for (int i = 0; i < count; ++i) {
    streams.push_back(c10::mcpu::getStreamFromPool(i / 2));
    (void)streams.back().stream();
  }
  std::atomic<int64_t> ran{0}, destroyed{0};
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::array<std::exception_ptr, count> errors{};
  std::vector<std::thread> threads;
  struct State {
    explicit State(std::atomic<int64_t>* counter) : destroyed(counter) {}
    ~State() {
      ++*destroyed;
    }
    std::atomic<int64_t>* destroyed;
    bool enabled = false;
  };
  for (int i = 0; i < count; ++i) {
    threads.emplace_back([&, i] {
      ++ready;
      while (!go.load())
        std::this_thread::yield();
      try {
        c10::mcpu::setCurrentMcpuStream(streams[i]);
        auto state = std::make_shared<State>(&destroyed);
        at::mcpu::KernelLaunchLifetimeScope lifetime(state);
        auto* ptr = state.get();
        at::mcpu::launch_kernel([ptr, i] { ptr->enabled = i % 2 == 0; });
        at::mcpu::KernelLaunchConditionScope condition(lifetime, &ptr->enabled);
        for (int64_t j = 0; j < launches; ++j) {
          MCPU_LAUNCH_TIMED_KERNEL(
              "mcpu::parallel_condition_test", ([&ran]), { ++ran; });
        }
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  while (ready.load() != count)
    std::this_thread::yield();
  go.store(true);
  for (auto& thread : threads)
    thread.join();
  for (auto stream : streams)
    stream.synchronize();
  for (auto error : errors)
    if (error)
      std::rethrow_exception(error);
  return {ran.load(), destroyed.load()};
}

std::tuple<int64_t, int64_t, int64_t> kernel_launch_lifetime_counts(
    const at::Tensor& stream_selector,
    int64_t launches) {
  check_stream_test_tensor(stream_selector);
  TORCH_CHECK(launches >= 0, "launches must be non-negative");

  g_lifetime_payload_constructed.store(0, std::memory_order_relaxed);
  g_lifetime_payload_destroyed.store(0, std::memory_order_relaxed);
  g_lifetime_payload_ran.store(0, std::memory_order_relaxed);

  for (int64_t i = 0; i < launches; ++i) {
    auto payload = std::make_unique<KernelLaunchLifetimePayload>(i);
    at::mcpu::launch_timed_kernel(
        "mcpu::kernel_launch_lifetime_counts",
        [payload = std::move(payload)](
            at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::kernel_launch_lifetime_counts", timing_event);
          TORCH_CHECK(payload != nullptr, "lifetime payload was not captured");
          g_lifetime_payload_ran.fetch_add(1, std::memory_order_relaxed);
          payload->value += 1;
        });
  }

  c10::mcpu::getCurrentMcpuStream(stream_selector.device().index())
      .synchronize();

  return std::make_tuple(
      g_lifetime_payload_constructed.load(std::memory_order_relaxed),
      g_lifetime_payload_destroyed.load(std::memory_order_relaxed),
      g_lifetime_payload_ran.load(std::memory_order_relaxed));
}

std::tuple<int64_t, int64_t, int64_t> kernel_launch_failed_lifetime_counts(
    const at::Tensor& stream_selector) {
  check_stream_test_tensor(stream_selector);

  g_lifetime_payload_constructed.store(0, std::memory_order_relaxed);
  g_lifetime_payload_destroyed.store(0, std::memory_order_relaxed);
  g_lifetime_payload_ran.store(0, std::memory_order_relaxed);

  bool caught_launch_failure = false;
  try {
    auto payload = std::make_unique<KernelLaunchLifetimePayload>(0);
    at::mcpu::launch_timed_kernel_on_stream(
        static_cast<orStream_t>(nullptr),
        "mcpu::kernel_launch_failed_lifetime_counts",
        [payload = std::move(payload)](
            at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::kernel_launch_failed_lifetime_counts", timing_event);
          g_lifetime_payload_ran.fetch_add(1, std::memory_order_relaxed);
        });
  } catch (const c10::Error&) {
    caught_launch_failure = true;
  }

  TORCH_CHECK(caught_launch_failure, "expected null stream launch to fail");

  return std::make_tuple(
      g_lifetime_payload_constructed.load(std::memory_order_relaxed),
      g_lifetime_payload_destroyed.load(std::memory_order_relaxed),
      g_lifetime_payload_ran.load(std::memory_order_relaxed));
}

} // namespace at::native::mcpu
