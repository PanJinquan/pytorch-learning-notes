#ifndef CAFFE2_OPERATORS_PACK_SEGMENTS_H_
#define CAFFE2_OPERATORS_PACK_SEGMENTS_H_

#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class PackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  PackSegmentsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        max_length_(OperatorBase::GetSingleArgument<int>("max_length", -1)),
        pad_minf_(OperatorBase::GetSingleArgument<bool>("pad_minf", false)),
        return_presence_mask_(OperatorBase::GetSingleArgument<bool>(
            "return_presence_mask",
            false)) {
    if (pad_minf_) {
      padding_ = -1.0 * std::numeric_limits<float>::infinity();
    } else {
      padding_ = 0;
    }
  }

  bool RunOnDevice() {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType();

  template <typename T, typename Data_T>
  bool DoRunWithType2();

  INPUT_TAGS(LENGTHS, DATA);

 private:
  TIndex max_length_;
  bool pad_minf_;
  float padding_;
  bool return_presence_mask_;

  // Scratch space required by the CUDA version
  Tensor dev_buffer_{Context::GetDeviceType()};
  Tensor dev_lengths_prefix_sum_{Context::GetDeviceType()};
  Tensor dev_max_length_{Context::GetDeviceType()};
  Tensor host_max_length_{CPU};
};

template <class Context>
class UnpackSegmentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  UnpackSegmentsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        max_length_(OperatorBase::GetSingleArgument<int>("max_length", -1)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
  }

  template <typename T>
  bool DoRunWithType();

  template <typename T, typename Data_T>
  bool DoRunWithType2();

  INPUT_TAGS(LENGTHS, DATA);

 private:
  TIndex max_length_;
  Tensor dev_buffer_{Context::GetDeviceType()};
  Tensor dev_lengths_prefix_sum_{Context::GetDeviceType()};
  Tensor dev_max_length_{Context::GetDeviceType()};
  Tensor dev_num_cell_{Context::GetDeviceType()};
  Tensor host_max_length_{CPU};
  Tensor host_num_cell_{CPU};
};

} // namespace caffe2
#endif // CAFFE2_OPERATORS_PACK_SEGMENTS_H_
