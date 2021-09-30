#ifndef CAFFE2_OPERATORS_ELEMENTWISE_SUB_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_SUB_OP_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct SubFunctor {
  template <typename TIn, typename TOut>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A,
      const TIn* B,
      TOut* C,
      Context* context) const {
    math::Sub(
        A_dims.size(),
        A_dims.data(),
        B_dims.size(),
        B_dims.data(),
        A,
        B,
        C,
        context);
    return true;
  }

  template <typename TGrad, typename TIn, typename TOut>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC,
      const TIn* /* A */,
      const TIn* /* B */,
      const TOut* /* C */,
      TGrad* dA,
      TGrad* dB,
      Context* context) const {
    const std::vector<int> C_dims =
        elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            A_dims, B_dims);
    std::vector<int> A_axes;
    std::vector<int> B_axes;
    elementwise_ops_utils::ComputeBinaryBroadcastBackwardAxes(
        A_dims, B_dims, &A_axes, &B_axes);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        A_axes.size(),
        A_axes.data(),
        dC,
        dA,
        context);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        B_axes.size(),
        B_axes.data(),
        dC,
        dB,
        context);
    const int size = std::accumulate(
        B_dims.cbegin(), B_dims.cend(), 1, std::multiplies<int>());
    math::Neg(size, dB, dB, context);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_SUB_OP_H_
