
#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/core/TensorBody.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/ATen.h>

#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/blob.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/index.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/view_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/view_native.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>

#include <memory>
#include <vector>

#include "../../aten/McpuTensorView.hpp"
#include "../../runtime/McpuKernelLaunch.h"

namespace torch::aot_inductor {

class AOTIMcpuGuard {
 public:
  explicit AOTIMcpuGuard(int32_t device_index)
      : guard_(c10::Device(c10::DeviceType::PrivateUse1, device_index)) {}

  void set_index(int32_t device_index) {
    guard_.set_index(device_index);
  }

 private:
  c10::DeviceGuard guard_;
};

class AOTIMcpuStreamGuard {
 public:
  AOTIMcpuStreamGuard(orStream_t stream, int32_t device_index)
      : guard_(
            c10::mcpu::getStreamFromExternal(stream, device_index).unwrap()) {}

 private:
  c10::OptionalStreamGuard guard_;
};

} // namespace torch::aot_inductor

using namespace torch::aot_inductor;

// Warning: 确保所有 op 没有返回新的
// Tensor，而是通过输出参数（Tensor(a!)）就地修改输入 Tensor。

using at::mcpu::get_cpu_tensor_view_if_needed;
using at::mcpu::get_cpu_view_from_mcpu_tensor;

namespace torch_mcpu_cpp_wrapper_detail {

struct TensorMeta {
  void* ptr;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  at::TensorOptions options;
};

inline TensorMeta make_tensor_meta(const at::Tensor& tensor) {
  return TensorMeta{
      tensor.data_ptr(),
      tensor.sizes().vec(),
      tensor.strides().vec(),
      tensor.options().device(c10::DeviceType::CPU)};
}

inline at::Tensor tensor_from_meta(const TensorMeta& meta) {
  return at::from_blob(meta.ptr, meta.sizes, meta.strides, meta.options);
}

} // namespace torch_mcpu_cpp_wrapper_detail

inline at::Tensor empty_unary_mcpu_result(
    const at::Tensor& self,
    at::ScalarType result_dtype) {
  return at::empty_like(
      self, self.options().dtype(result_dtype), at::MemoryFormat::Preserve);
}

inline at::Tensor empty_binary_mcpu_result(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  const bool self_is_mcpu =
      self.device().type() == c10::DeviceType::PrivateUse1;
  const bool other_is_mcpu =
      other.device().type() == c10::DeviceType::PrivateUse1;
  const at::Tensor& device_tensor = self_is_mcpu ? self : other;
  auto options = (self_is_mcpu || other_is_mcpu)
      ? device_tensor.options()
      : self.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0));
  return at::empty(out_sizes, options.dtype(at::result_type(self, other)));
}

inline std::vector<int64_t> cat_mcpu_result_sizes(
    const std::vector<at::Tensor>& tensors,
    int64_t dim) {
  TORCH_CHECK(
      !tensors.empty(),
      "aoti_torch_mcpu_cat: expected a non-empty tensor list");

  auto is_skippable_empty = [](const at::Tensor& tensor) {
    return tensor.dim() == 1 && tensor.numel() == 0;
  };

  auto first_nonempty = tensors.end();
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (!is_skippable_empty(*it)) {
      first_nonempty = it;
      break;
    }
  }

  if (first_nonempty == tensors.end()) {
    return {0};
  }

  const auto ndim = first_nonempty->dim();
  const auto wrapped_dim = at::maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> result(
      first_nonempty->sizes().begin(), first_nonempty->sizes().end());
  result[wrapped_dim] = 0;

  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    if (is_skippable_empty(tensor)) {
      continue;
    }
    TORCH_CHECK(
        tensor.dim() == ndim,
        "aoti_torch_mcpu_cat: expected all tensors to have the same number of dimensions");
    for (const auto d : c10::irange(ndim)) {
      if (d == static_cast<size_t>(wrapped_dim)) {
        continue;
      }
      TORCH_CHECK(
          tensor.size(d) == result[d],
          "aoti_torch_mcpu_cat: expected tensor sizes to match except in dimension ",
          wrapped_dim);
    }
    result[wrapped_dim] += tensor.size(wrapped_dim);
  }
  return result;
}

inline at::Tensor empty_cat_mcpu_result(
    const std::vector<at::Tensor>& tensors,
    int64_t dim) {
  auto out_sizes = cat_mcpu_result_sizes(tensors, dim);
  const auto* device_tensor = &tensors[0];
  for (const auto& tensor : tensors) {
    if (tensor.device().type() == c10::DeviceType::PrivateUse1) {
      device_tensor = &tensor;
      break;
    }
  }
  auto options = device_tensor->device().type() == c10::DeviceType::PrivateUse1
      ? device_tensor->options()
      : device_tensor->options().device(
            c10::Device(c10::DeviceType::PrivateUse1, 0));
  return at::empty(out_sizes, options);
}

inline std::vector<int64_t> embedding_mcpu_result_sizes(
    const at::Tensor& weight,
    const at::Tensor& indices) {
  TORCH_CHECK(
      weight.dim() >= 1,
      "aoti_torch_mcpu_embedding: expected weight to have at least one dimension");
  auto out_sizes = indices.sizes().vec();
  for (const auto d : c10::irange(1, weight.dim())) {
    out_sizes.push_back(weight.size(d));
  }
  return out_sizes;
}

inline at::Tensor empty_embedding_mcpu_result(
    const at::Tensor& weight,
    const at::Tensor& indices) {
  const bool weight_is_mcpu =
      weight.device().type() == c10::DeviceType::PrivateUse1;
  const bool indices_is_mcpu =
      indices.device().type() == c10::DeviceType::PrivateUse1;
  const at::Tensor& device_tensor =
      weight_is_mcpu ? weight : (indices_is_mcpu ? indices : weight);
  auto options = (weight_is_mcpu || indices_is_mcpu)
      ? device_tensor.options()
      : device_tensor.options().device(
            c10::Device(c10::DeviceType::PrivateUse1, 0));
  return at::empty(
      embedding_mcpu_result_sizes(weight, indices),
      options.dtype(weight.scalar_type()));
}

// clang-format off
inline AOTITorchError aoti_torch_mcpu_view_dtype(AtenTensorHandle self, int32_t dtype, AtenTensorHandle* ret0) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto tmp_result = at::compositeexplicitautograd::view(
            resolve_tensor_dispatch_flags(self), static_cast<c10::ScalarType>(dtype)
        );
        *ret0 = new_tensor_handle(std::move(tmp_result));
    });
}

inline AOTITorchError aoti_torch_mcpu_slice_Tensor(
    AtenTensorHandle self,
    int64_t dim,
    int64_t* start,
    int64_t* end,
    int64_t step,
    AtenTensorHandle* ret0) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        const auto start_value = start == nullptr
            ? std::optional<int64_t>()
            : std::optional<int64_t>(*start);
        const auto end_value = end == nullptr
            ? std::optional<int64_t>()
            : std::optional<int64_t>(*end);
        auto tmp_result = at::slice(
            resolve_tensor_dispatch_flags(self),
            dim,
            start_value,
            end_value,
            step);
        *ret0 = new_tensor_handle(std::move(tmp_result));
    });
}

inline AOTITorchError aoti_torch_mcpu_mm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2) {
    at::Tensor* t_out = tensor_handle_to_tensor_pointer(out);
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_mat2 = tensor_handle_to_tensor_pointer(mat2);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto self_meta = torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_self);
        auto mat2_meta = torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_mat2);
        auto out_meta = torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_out);
        at::mcpu::launch_kernel(*t_out, [self_meta = std::move(self_meta), mat2_meta = std::move(mat2_meta), out_meta = std::move(out_meta)]() mutable {
            at::mcpu::KernelPointerMemoryGuard guard({self_meta.ptr, mat2_meta.ptr, out_meta.ptr});
            at::Tensor cpu_out = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(out_meta);
            at::Tensor cpu_self = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(self_meta);
            at::Tensor cpu_mat2 = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(mat2_meta);
            at::mm_out(cpu_out, cpu_self, cpu_mat2);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_addmm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat1, AtenTensorHandle mat2, double beta, double alpha) {
    at::Tensor* t_out = tensor_handle_to_tensor_pointer(out);
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_mat1 = tensor_handle_to_tensor_pointer(mat1);
    at::Tensor* t_mat2 = tensor_handle_to_tensor_pointer(mat2);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        struct AddmmArgs {
            torch_mcpu_cpp_wrapper_detail::TensorMeta self;
            torch_mcpu_cpp_wrapper_detail::TensorMeta mat1;
            torch_mcpu_cpp_wrapper_detail::TensorMeta mat2;
            torch_mcpu_cpp_wrapper_detail::TensorMeta out;
            double beta;
            double alpha;
        };
        auto args = std::make_unique<AddmmArgs>(AddmmArgs{
            torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_self),
            torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_mat1),
            torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_mat2),
            torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_out),
            beta,
            alpha});
        at::mcpu::launch_kernel(*t_out, [args = std::move(args)]() mutable {
            at::mcpu::KernelPointerMemoryGuard guard({args->self.ptr, args->mat1.ptr, args->mat2.ptr, args->out.ptr});
            at::Tensor cpu_out = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(args->out);
            at::Tensor cpu_self = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(args->self);
            at::Tensor cpu_mat1 = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(args->mat1);
            at::Tensor cpu_mat2 = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(args->mat2);
            at::addmm_out(cpu_out, cpu_self, cpu_mat1, cpu_mat2, args->beta, args->alpha);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_add_Tensor(AtenTensorHandle self, AtenTensorHandle other, double alpha, AtenTensorHandle* ret0) {
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_other = tensor_handle_to_tensor_pointer(other);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto out = empty_binary_mcpu_result(*t_self, *t_other);
        auto launch_out = out;
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [out = launch_out, self = *t_self, other = *t_other, alpha]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self, other);
            at::Tensor cpu_self = get_cpu_tensor_view_if_needed(self);
            at::Tensor cpu_other = get_cpu_tensor_view_if_needed(other);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::add_out(cpu_out, cpu_self, cpu_other, alpha);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_mul_Tensor(AtenTensorHandle self, AtenTensorHandle other, AtenTensorHandle* ret0) {
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_other = tensor_handle_to_tensor_pointer(other);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto out = empty_binary_mcpu_result(*t_self, *t_other);
        auto launch_out = out;
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [out = launch_out, self = *t_self, other = *t_other]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self, other);
            at::Tensor cpu_self = get_cpu_tensor_view_if_needed(self);
            at::Tensor cpu_other = get_cpu_tensor_view_if_needed(other);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::mul_out(cpu_out, cpu_self, cpu_other);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_cat(const AtenTensorHandle* tensors, int64_t tensors_len_, int64_t dim, AtenTensorHandle* ret0) {
    std::vector<at::Tensor> tensor_vec;
    tensor_vec.reserve(tensors_len_);
    for (int64_t i = 0; i < tensors_len_; ++i) {
        tensor_vec.push_back(*tensor_handle_to_tensor_pointer(tensors[i]));
    }
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto out = empty_cat_mcpu_result(tensor_vec, dim);
        auto launch_out = out;
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [out = launch_out, tensor_vec, dim]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, c10::IValue(tensor_vec));
            std::vector<at::Tensor> cpu_tensors;
            cpu_tensors.reserve(tensor_vec.size());
            for (const auto& tensor : tensor_vec) {
                cpu_tensors.push_back(get_cpu_tensor_view_if_needed(tensor));
            }
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::cat_out(cpu_out, at::ITensorListRef(cpu_tensors), dim);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_constant_pad_nd(
    AtenTensorHandle self,
    const int64_t* pad,
    int64_t pad_len,
    double value,
    AtenTensorHandle* ret0) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto tmp_result = at::constant_pad_nd(
            resolve_tensor_dispatch_flags(self),
            at::IntArrayRef(pad, pad_len),
            value);
        *ret0 = new_tensor_handle(std::move(tmp_result));
    });
}

inline AOTITorchError aoti_torch_mcpu_embedding(AtenTensorHandle weight, AtenTensorHandle indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse, AtenTensorHandle* ret0) {
    at::Tensor* t_weight = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* t_indices = tensor_handle_to_tensor_pointer(indices);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        struct EmbeddingArgs {
            torch_mcpu_cpp_wrapper_detail::TensorMeta weight;
            torch_mcpu_cpp_wrapper_detail::TensorMeta indices;
            torch_mcpu_cpp_wrapper_detail::TensorMeta out;
            int64_t padding_idx;
            bool scale_grad_by_freq;
            bool sparse;
        };
        auto out = empty_embedding_mcpu_result(*t_weight, *t_indices);
        auto launch_out = out;
        auto args = std::make_unique<EmbeddingArgs>(EmbeddingArgs{
            torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_weight),
            torch_mcpu_cpp_wrapper_detail::make_tensor_meta(*t_indices),
            torch_mcpu_cpp_wrapper_detail::make_tensor_meta(launch_out),
            padding_idx,
            scale_grad_by_freq,
            sparse});
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [args = std::move(args)]() mutable {
            at::mcpu::KernelPointerMemoryGuard guard({args->weight.ptr, args->indices.ptr, args->out.ptr});
            at::Tensor cpu_weight = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(args->weight);
            at::Tensor cpu_indices = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(args->indices);
            at::Tensor cpu_out = torch_mcpu_cpp_wrapper_detail::tensor_from_meta(args->out);
            at::embedding_out(cpu_out, cpu_weight, cpu_indices, args->padding_idx, args->scale_grad_by_freq, args->sparse);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_index_Tensor(AtenTensorHandle self, const AtenTensorHandle** indices, int64_t indices_len_, AtenTensorHandle* ret0) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto resolved_indices =
            resolve_tensor_list_dispatch_flags(indices, indices_len_);
        c10::List<std::optional<at::Tensor>> index_list;
        index_list.reserve(resolved_indices.size());
        for (auto& index : resolved_indices) {
            index_list.push_back(std::move(index));
        }
        auto tmp_result = at::index(
            resolve_tensor_dispatch_flags(self), index_list);
        *ret0 = new_tensor_handle(std::move(tmp_result));
    });
}

inline AOTITorchError aoti_torch_mcpu_sigmoid(AtenTensorHandle self, AtenTensorHandle* ret0) {
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto result_dtype =
            c10::isIntegralType(t_self->scalar_type(), /*includeBool=*/true)
            ? at::kFloat
            : t_self->scalar_type();
        auto out = empty_unary_mcpu_result(*t_self, result_dtype);
        auto launch_out = out;
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [out = launch_out, self = *t_self]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self);
            at::Tensor cpu_self = get_cpu_view_from_mcpu_tensor(self);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::sigmoid_out(cpu_out, cpu_self);
        });
    });
}
// clang-format on
