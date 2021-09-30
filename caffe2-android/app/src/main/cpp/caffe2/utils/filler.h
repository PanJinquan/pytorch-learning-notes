#ifndef CAFFE2_FILLER_H_
#define CAFFE2_FILLER_H_

#include <sstream>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context_t>
class TensorFiller {
 public:
  template <class Type>
  void Fill(Tensor* tensor) const {
    CAFFE_ENFORCE(context_, "context is null");
    CAFFE_ENFORCE(tensor, "tensor is null");
    auto min = static_cast<Type>(min_);
    auto max = static_cast<Type>(max_);
    CAFFE_ENFORCE_LE(min, max);

    Tensor temp_tensor(shape_, Context_t::GetDeviceType());
    tensor->swap(temp_tensor);
    Type* data = tensor->template mutable_data<Type>();
    Context_t* context = static_cast<Context_t*>(context_);

    // TODO: Come up with a good distribution abstraction so that
    // the users could plug in their own distribution.
    if (has_fixed_sum_) {
      auto fixed_sum = static_cast<Type>(fixed_sum_);
      CAFFE_ENFORCE_LE(min * tensor->size(), fixed_sum);
      CAFFE_ENFORCE_GE(max * tensor->size(), fixed_sum);
      math::RandFixedSum<Type, Context_t>(
          tensor->size(), min, max, fixed_sum_, data, context);
    } else {
      math::RandUniform<Type, Context_t>(
          tensor->size(), min, max, data, context);
    }
  }

  template <class Type>
  TensorFiller& Min(Type min) {
    min_ = (double)min;
    return *this;
  }

  template <class Type>
  TensorFiller& Max(Type max) {
    max_ = (double)max;
    return *this;
  }

  template <class Type>
  TensorFiller& FixedSum(Type fixed_sum) {
    has_fixed_sum_ = true;
    fixed_sum_ = (double)fixed_sum;
    return *this;
  }

  // a helper function to construct the lengths vector for sparse features
  template <class Type>
  TensorFiller& SparseLengths(Type total_length) {
    return FixedSum(total_length).Min(0).Max(total_length);
  }

  // a helper function to construct the segments vector for sparse features
  template <class Type>
  TensorFiller& SparseSegments(Type max_segment) {
    CAFFE_ENFORCE(!has_fixed_sum_);
    return Min(0).Max(max_segment);
  }

  TensorFiller& Shape(const std::vector<TIndex>& shape) {
    shape_ = shape;
    return *this;
  }

  // Use new context so that it is independent from its operator
  TensorFiller& Context(Context_t* context) {
    context_ = (void*)context;
    return *this;
  }

  template <class Type>
  TensorFiller(
      const std::vector<TIndex>& shape,
      Type fixed_sum,
      Context_t* context)
      : shape_(shape),
        has_fixed_sum_(true),
        fixed_sum_((double)fixed_sum),
        context_((void*)context) {}

  TensorFiller(const std::vector<TIndex>& shape, Context_t* context)
      : shape_(shape),
        has_fixed_sum_(false),
        fixed_sum_(0),
        context_((void*)context) {}

  TensorFiller() : TensorFiller({}, (Context_t*)nullptr) {}

  std::string DebugString() const {
    std::stringstream stream;
    stream << "shape = [" << shape_ << "]; min = " << min_
           << "; max = " << max_;
    if (has_fixed_sum_) {
      stream << "; fixed sum = " << fixed_sum_;
    }
    return stream.str();
  }

 private:
  std::vector<TIndex> shape_;
  // TODO: type is unknown until a user starts to fill data;
  // cast everything to double for now.
  double min_ = 0.0;
  double max_ = 1.0;
  bool has_fixed_sum_;
  double fixed_sum_;
  void* context_;
};

} // namespace caffe2

#endif // CAFFE2_FILLER_H_
