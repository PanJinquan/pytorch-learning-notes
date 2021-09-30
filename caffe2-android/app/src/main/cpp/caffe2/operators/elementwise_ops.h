#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OPS_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OPS_H_

#include <iterator>
#include <string>
#include <tuple>
#include <vector>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

using NumericTypes = TensorTypes<int32_t, int64_t, float, double>;
using IntTypes = TensorTypes<int32_t, int64_t>;
using BoolTypes = TensorTypes<bool>;
using IntBoolTypes = TensorTypes<int32_t, int64_t, bool>; // discrete types

struct SameTypeAsInput {
  template <typename T>
  using type = T;
};

template <typename R>
struct FixedType {
  template <typename T>
  using type = R;
};

template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput>
class UnaryElementwiseWithArgsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  UnaryElementwiseWithArgsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), functor_(*this) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    return functor_(
        X.size(),
        X.template data<T>(),
        Y->template mutable_data<typename OutputTypeMap::template type<T>>(),
        &context_);
  }

 private:
  Functor functor_;
};

// UnaryFunctorWithDefaultCtor is a functor that can be used as the functor of
// an UnaryElementwiseWithArgsOp. It simply forwards the operator() call into
// another functor that doesn't accept arguments in its constructor.
template <class Functor>
struct UnaryFunctorWithDefaultCtor {
  explicit UnaryFunctorWithDefaultCtor(OperatorBase& /* op */) {}

  template <typename TIn, typename TOut, class Context>
  bool operator()(const int size, const TIn* X, TOut* Y, Context* context)
      const {
    return functor(size, X, Y, context);
  }

  Functor functor{};
};

// UnaryElementwiseOp is a wrapper around UnaryElementwiseWithArgsOp, with the
// difference that it takes a functor with default constructor, e.g. that does
// not need to take into consideration any arguments during operator creation.
template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput>
using UnaryElementwiseOp = UnaryElementwiseWithArgsOp<
    InputTypes,
    Context,
    UnaryFunctorWithDefaultCtor<Functor>,
    OutputTypeMap>;

template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput>
class BinaryElementwiseWithArgsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BinaryElementwiseWithArgsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW"),
        functor_(*this) {
    if (legacy_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1, "Unsupported axis string", axis_str_);
        const size_t semantic_axis_ = order_.find(axis_str_);
        CAFFE_ENFORCE_NE(
            semantic_axis_,
            string::npos,
            "Unrecognizable axis string ",
            axis_str_,
            " from order string ",
            order_);
        axis_ = semantic_axis_;
      } else {
        CAFFE_ENFORCE(
            axis_ == -1 && axis_str_.empty(),
            "Do not specify axis or axis_str if broadcast is not enabled.");
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* C = Output(0);
    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();
    std::vector<int> A_dims;
    std::vector<int> B_dims;

    if (legacy_broadcast_) {
      CAFFE_ENFORCE_NE(
          C,
          &B,
          "In-place is allowed only with the first tensor when "
          "legacy-broadcasting");
      C->ResizeLike(A);
      if (B.size() == 1) {
        A_dims = {static_cast<int>(A.size())};
        B_dims = {1};
      } else {
        size_t pre, n, post;
        std::tie(pre, n, post) =
            elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
        A_dims = {
            static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
        B_dims = {static_cast<int>(n), 1};
      }
    } else {
      std::copy(A.dims().cbegin(), A.dims().cend(), std::back_inserter(A_dims));
      std::copy(B.dims().cbegin(), B.dims().cend(), std::back_inserter(B_dims));
      const std::vector<int> C_dims =
          elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
              A_dims, B_dims);
      if (C == &A) {
        CAFFE_ENFORCE_EQ(C_dims, A_dims);
      } else if (C == &B) {
        CAFFE_ENFORCE_EQ(C_dims, B_dims);
      } else {
        C->Resize(C_dims);
      }
    }
    auto* C_data =
        C->template mutable_data<typename OutputTypeMap::template type<T>>();
    return functor_.Forward(A_dims, B_dims, A_data, B_data, C_data, &context_);
  }

 private:
  const bool legacy_broadcast_;
  int axis_;
  const std::string axis_str_;
  const std::string order_;

  Functor functor_;
};

template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput,
    class GradientTypeMap = SameTypeAsInput>
class BinaryElementwiseWithArgsGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BinaryElementwiseWithArgsGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW"),
        functor_(*this) {
    if (legacy_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1, "Unsupported axis string", axis_str_);
        const size_t semantic_axis_ = order_.find(axis_str_);
        CAFFE_ENFORCE_NE(
            semantic_axis_,
            string::npos,
            "Unrecognizable axis string ",
            axis_str_,
            " from order string ",
            order_);
        axis_ = semantic_axis_;
      } else {
        CAFFE_ENFORCE(
            axis_ == -1 && axis_str_.empty(),
            "Do not specify axis or axis_str if broadcast is not enabled.");
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dC = Input(0);
    const auto& A = Input(1);
    const auto& B = Input(2);
    auto* dA = Output(0);
    auto* dB = Output(1);
    vector<int> A_dims;
    vector<int> B_dims;
    if (legacy_broadcast_) {
      if (B.size() == 1) {
        A_dims = {static_cast<int>(A.size())};
        B_dims = {1};
      } else {
        size_t pre, n, post;
        std::tie(pre, n, post) =
            elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
        A_dims = {
            static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
        B_dims = {static_cast<int>(n), 1};
      }
    } else {
      std::copy(A.dims().cbegin(), A.dims().cend(), std::back_inserter(A_dims));
      std::copy(B.dims().cbegin(), B.dims().cend(), std::back_inserter(B_dims));
    }
    const typename OutputTypeMap::template type<T>* C_data = nullptr;
    if (InputSize() == 4) {
      const auto& C = Input(3);
      C_data = C.template data<typename OutputTypeMap::template type<T>>();
    }
    const auto* dC_data =
        dC.template data<typename GradientTypeMap::template type<T>>();
    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();
    dA->ResizeLike(A);
    dB->ResizeLike(B);
    auto* dA_data =
        dA->template mutable_data<typename GradientTypeMap::template type<T>>();
    auto* dB_data =
        dB->template mutable_data<typename GradientTypeMap::template type<T>>();
    return functor_.Backward(
        A_dims,
        B_dims,
        dC_data,
        A_data,
        B_data,
        C_data,
        dA_data,
        dB_data,
        &context_);
  }

 private:
  const bool legacy_broadcast_;
  int axis_;
  const std::string axis_str_;
  const std::string order_;

  Functor functor_;
};

template <class Functor>
struct BinaryFunctorWithDefaultCtor {
  explicit BinaryFunctorWithDefaultCtor(OperatorBase& /* op */) {}

  template <typename TIn, typename TOut, class Context>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A_data,
      const TIn* B_data,
      TOut* C_data,
      Context* context) const {
    return functor.Forward(A_dims, B_dims, A_data, B_data, C_data, context);
  }

  template <typename TGrad, typename TIn, typename TOut, class Context>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC_data,
      const TIn* A_data,
      const TIn* B_data,
      const TOut* C_data,
      TGrad* dA_data,
      TGrad* dB_data,
      Context* context) const {
    return functor.Backward(
        A_dims,
        B_dims,
        dC_data,
        A_data,
        B_data,
        C_data,
        dA_data,
        dB_data,
        context);
  }

  Functor functor{};
};

// BinaryElementwiseOp is a wrapper around BinaryElementwiseWithArgsOp, with the
// difference that it takes a functor with default constructor, e.g. that does
// not need to take into consideration any arguments during operator creation.
template <
    typename InputTypes,
    class Context,
    class Functor,
    class TypeMap = SameTypeAsInput>
using BinaryElementwiseOp = BinaryElementwiseWithArgsOp<
    InputTypes,
    Context,
    BinaryFunctorWithDefaultCtor<Functor>,
    TypeMap>;

// BinaryElementwiseGradientOp is a wrapper around
// BinaryElementwiseGradientWithArgsOp, with the difference that it takes a
// functor with default constructor, e.g. that does not need to take into
// consideration any arguments during operator creation.
template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput,
    class GradientTypeMap = SameTypeAsInput>
using BinaryElementwiseGradientOp = BinaryElementwiseWithArgsGradientOp<
    InputTypes,
    Context,
    BinaryFunctorWithDefaultCtor<Functor>,
    OutputTypeMap,
    GradientTypeMap>;

// Forward-only Unary Functors.
template <class Context>
struct NotFunctor {
  bool operator()(const int N, const bool* X, bool* Y, Context* context) const {
    math::Not(N, X, Y, context);
    return true;
  }
};

template <class Context>
struct SignFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    math::Sign(N, X, Y, context);
    return true;
  }
};

// Forward-only Binary Functors.
#define CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(FunctorName) \
  template <class Context>                                     \
  struct FunctorName##Functor {                                \
    template <typename TIn, typename TOut>                     \
    bool Forward(                                              \
        const std::vector<int>& A_dims,                        \
        const std::vector<int>& B_dims,                        \
        const TIn* A,                                          \
        const TIn* B,                                          \
        TOut* C,                                               \
        Context* context) const {                              \
      math::FunctorName(                                       \
          A_dims.size(),                                       \
          A_dims.data(),                                       \
          B_dims.size(),                                       \
          B_dims.data(),                                       \
          A,                                                   \
          B,                                                   \
          C,                                                   \
          context);                                            \
      return true;                                             \
    }                                                          \
  };

// Compare functors.
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(EQ);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(NE);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(LT);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(LE);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(GT);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(GE);

// Logical functors.
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(And);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(Or);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(Xor);

// Bitwise functors.
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(BitwiseAnd);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(BitwiseOr);
CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR(BitwiseXor);

#undef CAFFE2_DECLARE_FOWARD_ONLY_BINARY_FUNCTOR

namespace SRLHelper {

template <typename T>
void sum2one(const T* a, T* y, size_t n);

template <typename T>
void RunWithBroadcastFront(const T* a, T* y, size_t pre, size_t n, CPUContext*);

template <typename T>
void RunWithBroadcastBack(const T* a, T* y, size_t post, size_t n, CPUContext*);

template <typename T>
void RunWithBroadcast2(
    const T* a,
    T* y,
    size_t pre,
    size_t n,
    size_t post,
    CPUContext*);

} // namespace SRLHelper

// Sum reduction operator that is used for computing the gradient in cases
// where the forward op is in broadcast mode.
template <class Context>
class SumReduceLikeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SumReduceLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW") {
    if (axis_ != -1) {
      // Get axis from an explicit axis argument.
      CAFFE_ENFORCE_EQ(
          axis_str_.size(),
          0,
          "Args axis and axis_str cannot be used simultaneously.");
    } else if (axis_str_.size()) {
      // Get the axis index semantically.
      CAFFE_ENFORCE_EQ(
          axis_str_.size(), 1, "Unsupported axis string", axis_str_);
      size_t semantic_axis = order_.find(axis_str_);
      CAFFE_ENFORCE_NE(
          semantic_axis,
          string::npos,
          "Unrecognizable axis string ",
          axis_str_,
          " from order string ",
          order_);
      axis_ = semantic_axis;
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  int axis_;
  string axis_str_;
  string order_;
  Tensor ones_{Context::GetDeviceType()};
  Tensor sum_buffer_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_OPS_H_
