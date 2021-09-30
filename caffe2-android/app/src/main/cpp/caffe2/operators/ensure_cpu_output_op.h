#ifndef CAFFE2_OPERATORS_ENSURE_CPU_OUTPUT_OP_H_
#define CAFFE2_OPERATORS_ENSURE_CPU_OUTPUT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class EnsureCPUOutputOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  EnsureCPUOutputOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    if (OperatorBase::InputIsType<Tensor>(0, CPU)) {
      return CopyWithContext<CPUContext>();
    } else if (OperatorBase::InputIsType<Tensor>(0, Context::GetDeviceType())) {
      // CUDA Context will go this branch
      return CopyWithContext<Context>();
    } else {
      CAFFE_THROW(
          "Unexpected Input Blob: ",
          OperatorBase::Inputs().at(0)->meta().name());
    }
    return true;
  }

 private:
  template <class InputContext>
  bool CopyWithContext() {
    // Output is always on CPU
    auto* output = OperatorBase::Output<Tensor>(0, CPU);
    auto& input = OperatorBase::Input<Tensor>(0, InputContext::GetDeviceType());
    output->ResizeLike(input);
    context_.CopyItemsToCPU(
        input.meta(),
        input.size(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ENSURE_CPU_OUTPUT_OP_H_
