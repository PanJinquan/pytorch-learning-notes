#ifndef CAFFE2_OPERATORS_TRANSPOSE_H_
#define CAFFE2_OPERATORS_TRANSPOSE_H_

#include <algorithm>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class TransposeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  TransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {
    // We will check the legality of axes_: it should be from 0 to axes_.size().
    std::vector<int> axes_sorted = axes_;
    std::sort(axes_sorted.begin(), axes_sorted.end());
    for (std::size_t i = 0; i < axes_sorted.size(); ++i) {
      if (axes_sorted[i] != i) {
        CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
      }
    }
  }

  ~TransposeOp() = default;

  bool RunOnDevice() override {
    // Do the actual transpose, which is implemented in DoRunWithType().
    return DispatchHelper<TensorTypes<float, double, int, TIndex>>::call(
        this, Input(0));
  }

 private:
  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.ndim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.rbegin(), axes_.rend(), 0);
    } else {
      CAFFE_ENFORCE_EQ(ndim, axes_.size());
    }
    const std::vector<int> X_dims(X.dims().cbegin(), X.dims().cend());
    std::vector<int> Y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      Y_dims[i] = X_dims[axes_[i]];
    }
    Y->Resize(Y_dims);
    math::Transpose<T, Context>(
        X_dims.size(),
        X_dims.data(),
        axes_.data(),
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

  std::vector<int> axes_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TRANSPOSE_H_
