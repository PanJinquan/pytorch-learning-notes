#pragma once

#include <caffe2/core/common.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <caffe2/ideep/ideep_utils.h>
#include <caffe2/proto/caffe2.pb.h>

namespace caffe2 {

/**
 * @brief A templated class to allow one to wrap a CPU operator as an IDEEP
 * operator.
 *
 * This class can be used when one does not have the IDEEP implementation ready
 * yet for an operator. Essentially, what this op does is to automatically
 * deal with data copy for you. Plausibly, this causes a lot of overhead and
 * is not optimal, so you should use this operator mostly for quick prototyping
 * purpose.
 *
 * All the input and output of the original operator should be TensorCPU.
 *
 * Example usage: if you have a class MyMagicOp that is CPU based, and you use
 * the registration code
 *     REGISTER_CPU_OPERATOR(MyMagic, MyMagicOp);
 * to register the CPU side, you can create its corresponding IDEEP operator
 * (with performance hits of course) via
 *     REGISTER_IDEEP_OPERATOR(MyMagic,
 *                            IDEEPFallbackOp<MyMagicOp>);
 *
 * Advanced usage: if you want to have some specific outputs never copied, you
 * can use the SkipOutputCopy template argument to do that. For example, if
 * MyMagic produces two outputs and the first output is always going to live on
 * the CPU, you can do
 *     REGISTER_IDEEP_OPERATOR(MyMagic,
 *                            IDEEPFallbackOp<MyMagicOp, SkipIndices<0>>);
 */
template <class CPUOp, typename SkipOutputCopy = SkipIndices<>>
class IDEEPFallbackOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPFallbackOp(const OperatorDef& def, Workspace* ws)
      : IDEEPOperator(def, ws) {
    CAFFE_ENFORCE_EQ(def.device_option().device_type(), IDEEP);
    base_def_.CopyFrom(def);
    // base_def_ runs on CPU, so we will set its device option to CPU.
    // Copy to allow random_seed to be correctly propagated.
    base_def_.mutable_device_option()->CopyFrom(def.device_option());
    base_def_.mutable_device_option()->set_device_type(CPU);
    // Create output blobs in parent workspace,
    // then forward output blobs to local workspace.
    std::unordered_map<string, string> forwarded_output_blobs;
    for (int i = 0; i < base_def_.output_size(); i++) {
      string parent_name(base_def_.output(i));
      if (!SkipOutputCopy::Contains(i)) {
        parent_name += "_cpu_output_blob_" + base_def_.type();
      }
      local_output_blobs_.push_back(ws->CreateBlob(parent_name));
      CHECK_NOTNULL(local_output_blobs_.back());
      forwarded_output_blobs[base_def_.output(i)] = parent_name;
    }
    local_ws_.reset(new Workspace(ws, forwarded_output_blobs));
    // Set up the symbols for the local workspace.
    for (const string& name : base_def_.input()) {
      local_input_blobs_.push_back(local_ws_->CreateBlob(name));
      CHECK_NOTNULL(local_input_blobs_.back());
    }
    base_op_.reset(new CPUOp(base_def_, local_ws_.get()));
  }

  bool RunOnDevice() override {
    for (int i = 0; i < InputSize(); ++i) {
      if (InputIsType<itensor>(i) && Input(i).get_data_type() == itensor::data_type::f32) {
        auto& input = Input(i);
        auto dtensor = local_input_blobs_[i]->GetMutableTensor(CPU);
        dtensor->Resize(input.get_dims());
        if (input.is_public_format()) {
          dtensor->ShareExternalPointer(static_cast<float*>(input.get_data_handle()));
        } else {
          input.reorder_to(dtensor->template mutable_data<float>());
        }
      } else if (
          InputIsType<itensor>(i) &&
          Input(i).get_data_type() == itensor::data_type::s32) {
        auto& input = Input(i);
        auto dtensor = local_input_blobs_[i]->GetMutableTensor(CPU);
        dtensor->Resize(input.get_dims());
        if (input.is_public_format()) {
          dtensor->ShareExternalPointer(
              static_cast<long*>(input.get_data_handle()));
        } else {
          input.reorder_to(dtensor->template mutable_data<long>());
        }
      } else {
        VLOG(1) << "Input " << i << " is not ideep::tensor. Skipping copy.";
        // Note(jiayq): This removes a const but conceptually
        // local_input_blobs will only be used as const blob input for the
        // base op so we are still fine.
        local_input_blobs_[i]->ShareExternal(
            const_cast<void*>(OperatorBase::Inputs()[i]->GetRaw()),
            OperatorBase::Inputs()[i]->meta());
      }
    }

    if (!base_op_->Run()) {
      LOG(ERROR) << "Base op run failed in IDEEPFallbackOp. Def: "
                 << ProtoDebugString(this->debug_def());
      return false;
    }

    for (int i = 0; i < OutputSize(); ++i) {
      if (SkipOutputCopy::Contains(i)) {
        VLOG(1) << "Copy output: index " << i << " skipped.";
        continue;
      }
      CAFFE_ENFORCE(
          local_output_blobs_[i]->template IsType<TensorCPU>(),
          "IDEEP fallback op currently does not support non-TensorCPU "
          "output type who needs copying.");
      const auto& src = local_output_blobs_[i]->template Get<TensorCPU>();

      if (src.template IsType<float>()) {
        Blob* dst = OperatorBase::OutputBlob(i);
        if (!dst->template IsType<itensor>()) {
          dst->Reset(new itensor());
        }

        auto src_dims = src.dims();
        itensor::dims dst_dims (src_dims.begin(), src_dims.end());
        auto dtensor = dst->template GetMutable<itensor>();
        if (dtensor->get_dims() != dst_dims) {
          dtensor->resize(dst_dims, itensor::data_type::f32);
        }
        dtensor->set_data_handle(const_cast<void*>(src.raw_data()));
      } else {
        VLOG(2) << "Output " << base_def_.output(i) << " as CPUTensor";
        auto src_dims = src.dims();
        Blob* dst = OperatorBase::OutputBlob(i);
        dst->Reset(new Tensor(CPU));
        auto dtensor = dst->GetMutableTensor(CPU);
        dtensor->Resize(src_dims);
        dtensor->ShareData(src);
      }
    }
    return true;
  }

 protected:
  vector<Blob*> local_input_blobs_;
  vector<Blob*> local_output_blobs_;
  std::unique_ptr<CPUOp> base_op_;
  std::unique_ptr<Workspace> local_ws_;
  OperatorDef base_def_;
};

} // namespace caffe2
