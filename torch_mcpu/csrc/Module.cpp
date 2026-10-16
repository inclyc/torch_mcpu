#include <ATen/ATen.h>
#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include <c10/core/CachingDeviceAllocator.h>
#include <runtime/McpuKernelTiming.h>
#include <runtime/OpenRegFunctions.h>
#include <runtime/McpuOpTiming.h>
#include <runtime/OpenRegStream.h>
#include <runtime/OpenRegSharedAllocator.h>
#include <limits>

// Forward-declare allocator management functions from libtorch_mcpu.so.
// (DeviceCachingAllocator.h transitively includes openreg.h which is not on
//  torch_bindings include path, so we use explicit declarations instead.)
namespace c10::mcpu {
void emptyCache(c10::MempoolId_t mempool_id = {0, 0});
c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    c10::DeviceIndex device);
void resetPeakStats(c10::DeviceIndex device);
void resetAccumulatedStats(c10::DeviceIndex device);
} // namespace c10::mcpu

namespace c10d {
PyObject* _create_process_group_mcpu(PyObject* self, PyObject* args);
void registerMcpuDistributedBindings(PyObject* module);
}

namespace py = pybind11;

static PyObject* _xmalloc(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_arg = nullptr;
  int device_index = 0;
  if (!PyArg_ParseTuple(args, "Oi", &size_arg, &device_index)) {
    return nullptr;
  }
  const size_t size = PyLong_AsSize_t(size_arg);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  TORCH_CHECK(
      size > 0 &&
          size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "xmalloc size must be positive and fit in int64");
  torch::utils::device_lazy_init(at::kPrivateUse1);
  TORCH_CHECK(
      device_index >= 0 && device_index < c10::mcpu::device_count(),
      "invalid xmalloc device index");
  const c10::Device device(c10::DeviceType::PrivateUse1, device_index);
  auto tensor =
      at::for_blob(c10::mcpu::xmalloc(size), {static_cast<int64_t>(size)})
          .options(at::TensorOptions().dtype(at::kByte).device(device))
          .target_device(device)
          .make_tensor();
  // The arena owns the memory until process exit. Storage deletion must not
  // return it to the ordinary allocator, even if an async kernel still uses it.
  return THPVariable_Wrap(tensor);
  END_HANDLE_TH_ERRORS
}

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE MCPU GET DEFAULT GENERATOR
static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  return THPGenerator_initDefaultGenerator(at::globalContext().defaultGenerator(
      c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}
// LITERALINCLUDE MCPU GET DEFAULT GENERATOR

// LITERALINCLUDE START: MODULE SET DEVICE HELPER

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackDeviceIndex(arg);
  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::mcpu::set_device(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE END: MODULE SET DEVICE HELPER

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::mcpu::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(c10::mcpu::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(c10::mcpu::device_count());
  END_HANDLE_TH_ERRORS
}

static PyObject* _emptyCache(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::mcpu::emptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _memoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "_memory_stats expects a device index");
  auto device = THPUtils_unpackDeviceIndex(arg);
  auto stats = c10::mcpu::getDeviceStats(device);

  PyObject* dict = PyDict_New();
  auto insert = [&](const char* key, size_t val) {
    PyObject* v = PyLong_FromSize_t(val);
    PyDict_SetItemString(dict, key, v);
    Py_DECREF(v);
  };
  auto insert_stat = [&](const char* prefix,
                         const c10::CachingDeviceAllocator::Stat& s) {
    std::string cur = std::string(prefix) + ".current";
    std::string peak = std::string(prefix) + ".peak";
    std::string alloc = std::string(prefix) + ".allocated";
    std::string freed = std::string(prefix) + ".freed";
    insert(cur.c_str(), static_cast<size_t>(s.current));
    insert(peak.c_str(), static_cast<size_t>(s.peak));
    insert(alloc.c_str(), static_cast<size_t>(s.allocated));
    insert(freed.c_str(), static_cast<size_t>(s.freed));
  };
  insert_stat("allocated_bytes.all", stats.allocated_bytes[0]);
  insert_stat("reserved_bytes.all", stats.reserved_bytes[0]);
  insert_stat("active_bytes.all", stats.active_bytes[0]);
  insert("num_alloc_retries", static_cast<size_t>(stats.num_alloc_retries));
  return dict;
  END_HANDLE_TH_ERRORS
}

static PyObject* _resetPeakMemoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_reset_peak_memory_stats expects a device index");
  auto device = THPUtils_unpackDeviceIndex(arg);
  c10::mcpu::resetPeakStats(device);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _resetAccumulatedMemoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_reset_accumulated_memory_stats expects a device index");
  auto device = THPUtils_unpackDeviceIndex(arg);
  c10::mcpu::resetAccumulatedStats(device);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _getStreamPriorityRange(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  int least = 0, greatest = 0;
  c10::mcpu::getStreamPriorityRange(&least, &greatest);
  return Py_BuildValue("(ii)", least, greatest);
  END_HANDLE_TH_ERRORS
}

static PyObject* _getDefaultStream(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "_get_default_stream expects an int");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto stream = c10::mcpu::getDefaultMcpuStream(device_index).pack3();
  return Py_BuildValue(
      "(Lii)",
      static_cast<long long>(stream.stream_id),
      static_cast<int>(stream.device_index),
      static_cast<int>(stream.device_type));
  END_HANDLE_TH_ERRORS
}

static const char* idlePolicyName(orStream::IdlePolicy policy) {
  switch (policy) {
    case orStream::IdlePolicy::Busy:
      return "busy";
    case orStream::IdlePolicy::Hybrid:
      return "hybrid";
    case orStream::IdlePolicy::Block:
      return "block";
  }
  return "unknown";
}

static PyObject* _getStreamWorkerPolicy(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  long long stream_id = 0;
  int device_index = 0;
  int device_type = 0;
  if (!PyArg_ParseTuple(args, "Lii", &stream_id, &device_index, &device_type)) {
    return nullptr;
  }

  TORCH_CHECK(stream_id >= 0, "stream_id must be non-negative");
  TORCH_CHECK(
      static_cast<c10::DeviceType>(device_type) == c10::DeviceType::PrivateUse1,
      "stream device type must be mcpu/privateuseone");
  TORCH_CHECK(
      (static_cast<c10::StreamId>(stream_id) & 1) != 0,
      "external streams are not supported by _get_stream_worker_policy");

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto stream = c10::mcpu::McpuStream::unpack3(
      static_cast<c10::StreamId>(stream_id),
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));
  auto or_stream = stream.stream();
  return Py_BuildValue(
      "{s:i,s:i,s:s,s:i}",
      "priority",
      or_stream->priority,
      "is_default_stream",
      or_stream->is_default_stream ? 1 : 0,
      "idle_policy",
      idlePolicyName(or_stream->idle_policy),
      "affinity_core",
      or_stream->affinity_core);
  END_HANDLE_TH_ERRORS
}

static PyObject* _setOpTimingEnabled(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::mcpu::op_timing::set_enabled(PyObject_IsTrue(arg) != 0);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _resetOpTiming(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::mcpu::op_timing::reset();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _getOpTiming(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto snapshots = at::mcpu::op_timing::snapshot();
  PyObject* py_threads = PyList_New(static_cast<Py_ssize_t>(snapshots.size()));
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(snapshots.size()); ++i) {
    const auto& snapshot = snapshots[static_cast<std::size_t>(i)];
    PyObject* py_records =
        PyList_New(static_cast<Py_ssize_t>(snapshot.records.size()));
    for (Py_ssize_t j = 0; j < static_cast<Py_ssize_t>(snapshot.records.size());
         ++j) {
      const auto& record = snapshot.records[static_cast<std::size_t>(j)];
      PyObject* py_record = Py_BuildValue(
          "{s:s,s:s,s:K}",
          "op",
          record.op,
          "phase",
          record.phase,
          "t_ns",
          static_cast<unsigned long long>(record.t_ns));
      PyList_SET_ITEM(py_records, j, py_record);
    }
    PyObject* py_thread = Py_BuildValue(
        "{s:s,s:O}", "role", snapshot.role, "records", py_records);
    Py_DECREF(py_records);
    PyList_SET_ITEM(py_threads, i, py_thread);
  }
  return py_threads;
  END_HANDLE_TH_ERRORS
}

static PyObject* _setKernelTimingEnabled(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::mcpu::kernel_timing::set_enabled(PyObject_IsTrue(arg) != 0);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _resetKernelTiming(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::mcpu::kernel_timing::reset();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _getKernelTiming(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto snapshots = at::mcpu::kernel_timing::snapshot();
  PyObject* py_threads = PyList_New(static_cast<Py_ssize_t>(snapshots.size()));
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(snapshots.size()); ++i) {
    const auto& snapshot = snapshots[static_cast<std::size_t>(i)];
    PyObject* py_events =
        PyList_New(static_cast<Py_ssize_t>(snapshot.events.size()));
    for (Py_ssize_t j = 0; j < static_cast<Py_ssize_t>(snapshot.events.size());
         ++j) {
      const auto& event = snapshot.events[static_cast<std::size_t>(j)];
      const auto elapsed_time =
          event.end_time > event.begin_time ? event.end_time - event.begin_time
                                            : 0;
      PyObject* py_event = Py_BuildValue(
          "{s:s,s:K,s:K,s:K,s:K}",
          "name",
          event.name,
          "stream",
          static_cast<unsigned long long>(event.stream),
          "begin_time",
          static_cast<unsigned long long>(event.begin_time),
          "end_time",
          static_cast<unsigned long long>(event.end_time),
          "elapsed_time",
          static_cast<unsigned long long>(elapsed_time));
      PyList_SET_ITEM(py_events, j, py_event);
    }
    PyObject* py_thread = Py_BuildValue(
        "{s:s,s:O}", "role", snapshot.role, "events", py_events);
    Py_DECREF(py_events);
    PyList_SET_ITEM(py_threads, i, py_thread);
  }
  return py_threads;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE MCPU MODULE METHODS
static PyMethodDef methods[] = {
    {"_xmalloc", _xmalloc, METH_VARARGS, nullptr},
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {"_empty_cache", _emptyCache, METH_NOARGS, nullptr},
    {"_memory_stats", _memoryStats, METH_O, nullptr},
    {"_reset_peak_memory_stats", _resetPeakMemoryStats, METH_O, nullptr},
    {"_reset_accumulated_memory_stats",
     _resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_get_stream_priority_range",
     _getStreamPriorityRange,
     METH_NOARGS,
     nullptr},
    {"_get_default_stream", _getDefaultStream, METH_O, nullptr},
    {"_get_stream_worker_policy", _getStreamWorkerPolicy, METH_VARARGS, nullptr},
    {"_set_op_timing_enabled", _setOpTimingEnabled, METH_O, nullptr},
    {"_reset_op_timing", _resetOpTiming, METH_NOARGS, nullptr},
    {"_get_op_timing", _getOpTiming, METH_NOARGS, nullptr},
    {"_set_kernel_timing_enabled", _setKernelTimingEnabled, METH_O, nullptr},
    {"_reset_kernel_timing", _resetKernelTiming, METH_NOARGS, nullptr},
    {"_get_kernel_timing", _getKernelTiming, METH_NOARGS, nullptr},
    {"create_process_group_mcpu",
     c10d::_create_process_group_mcpu,
     METH_VARARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};
// LITERALINCLUDE MCPU MODULE METHODS
/*
 * When ASAN is enabled, PyTorch modifies the dlopen flag during import,
 * causing all global and weak symbols in _C.so and its dependent libraries
 * to be exposed to the global symbol scope, which in turn causes
 * subsequent symbols with the same name in other libraries to be intercepted.
 * Therefore, it cannot be named initModule here, otherwise initModule
 * in torch/csrc/Module.cpp will be called, resulting in failure.
 */
extern "C" MCPU_EXPORT PyObject* initMcpuModule(void) {
  static struct PyModuleDef mcpu_C_module = {
      PyModuleDef_HEAD_INIT, "torch_mcpu._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&mcpu_C_module);
  c10d::registerMcpuDistributedBindings(mod);

  return mod;
}
