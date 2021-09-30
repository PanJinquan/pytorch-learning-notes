#ifndef CAFFE2_OPERATORS_CONV_TRANSPOSE_UNPOOL_OP_BASE_H_
#define CAFFE2_OPERATORS_CONV_TRANSPOSE_UNPOOL_OP_BASE_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/proto/caffe2_legacy.pb.h"
#include "caffe2/utils/math.h"

CAFFE2_DECLARE_bool(caffe2_force_shared_col_buffer);

namespace caffe2 {

template <class Context>
class ConvTransposeUnpoolBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ConvTransposeUnpoolBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        legacy_pad_(
            static_cast<LegacyPadding>(OperatorBase::GetSingleArgument<int>(
                "legacy_pad",
                LegacyPadding::NOTSET))),
        kernel_(OperatorBase::GetRepeatedArgument<int>("kernels")),
        stride_(OperatorBase::GetRepeatedArgument<int>("strides")),
        pads_(OperatorBase::GetRepeatedArgument<int>("pads")),
        adj_(OperatorBase::GetRepeatedArgument<int>("adjs")),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        shared_buffer_(
            OperatorBase::GetSingleArgument<int>("shared_buffer", 0)),
        ws_(ws) {
    // For the padding, they should either be the legacy padding strategy
    // (VALID or SAME), or an explicit, non-negative value.
    if (legacy_pad_ == LegacyPadding::VALID ||
        legacy_pad_ == LegacyPadding::SAME) {
      CAFFE_ENFORCE(
          !OperatorBase::HasArgument("pads"),
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
    }
    // Get old arguments values.
    if (OperatorBase::HasArgument("kernel")) {
      kernel_.resize(2, OperatorBase::GetSingleArgument<int>("kernel", 0));
    } else if (
        OperatorBase::HasArgument("kernel_h") &&
        OperatorBase::HasArgument("kernel_w")) {
      kernel_.push_back(OperatorBase::GetSingleArgument<int>("kernel_h", 0));
      kernel_.push_back(OperatorBase::GetSingleArgument<int>("kernel_w", 0));
    }

    if (OperatorBase::HasArgument("stride")) {
      stride_.resize(2, OperatorBase::GetSingleArgument<int>("stride", 0));
    } else if (
        OperatorBase::HasArgument("stride_h") &&
        OperatorBase::HasArgument("stride_w")) {
      stride_.push_back(OperatorBase::GetSingleArgument<int>("stride_h", 0));
      stride_.push_back(OperatorBase::GetSingleArgument<int>("stride_w", 0));
    }

    if (OperatorBase::HasArgument("adj")) {
      adj_.resize(2, OperatorBase::GetSingleArgument<int>("adj", 0));
    } else if (
        OperatorBase::HasArgument("adj_h") &&
        OperatorBase::HasArgument("adj_w")) {
      adj_.push_back(OperatorBase::GetSingleArgument<int>("adj_h", 0));
      adj_.push_back(OperatorBase::GetSingleArgument<int>("adj_w", 0));
    }

    if (OperatorBase::HasArgument("pad")) {
      CAFFE_ENFORCE(
          legacy_pad_ != LegacyPadding::VALID &&
              legacy_pad_ != LegacyPadding::SAME,
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
      pads_.resize(4, OperatorBase::GetSingleArgument<int>("pad", 0));
    } else if (
        OperatorBase::HasArgument("pad_t") &&
        OperatorBase::HasArgument("pad_l") &&
        OperatorBase::HasArgument("pad_b") &&
        OperatorBase::HasArgument("pad_r")) {
      CAFFE_ENFORCE(
          legacy_pad_ != LegacyPadding::VALID &&
              legacy_pad_ != LegacyPadding::SAME,
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_t", 0));
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_l", 0));
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_b", 0));
      pads_.push_back(OperatorBase::GetSingleArgument<int>("pad_r", 0));
    }

    // Fill default values.
    if (kernel_.size() == 0) {
      kernel_.assign({0, 0});
    }

    if (stride_.size() == 0) {
      stride_.resize(kernel_.size(), 1);
    }

    if (pads_.size() == 0) {
      pads_.resize(kernel_.size() * 2, 0);
    }

    if (adj_.size() == 0) {
      adj_.resize(kernel_.size(), 0);
    }

    CAFFE_ENFORCE_EQ(stride_.size(), kernel_.size());
    CAFFE_ENFORCE_EQ(adj_.size(), kernel_.size());

    if (legacy_pad_ != LegacyPadding::VALID &&
        legacy_pad_ != LegacyPadding::SAME) {
      CAFFE_ENFORCE_EQ(pads_.size(), 2 * kernel_.size());
    }

    for (int dim = 0; dim < kernel_.size(); ++dim) {
      CAFFE_ENFORCE_GT(kernel_[dim], 0);
      CAFFE_ENFORCE_GT(stride_[dim], 0);
      CAFFE_ENFORCE_GE(adj_[dim], 0);
      CAFFE_ENFORCE_LE(adj_[dim], stride_[dim]);
    }

    // Create shared buffer mutex in the constructor
    // to avoid race-condition in DAGNet.
    if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
      createSharedBuffer<Context>(ws_);
    }
  }
  // Sets the output size. The output channel is manually specified.
  void SetOutputSize(const Tensor& input, Tensor* output, int output_channel) {
    CAFFE_ENFORCE(4 == input.ndim());
    CAFFE_ENFORCE(input.size() > 0);
    int N = input.dim32(0);
    bool channel_first = false; // initialized to suppress compiler warning.
    int H = 0, W = 0; // initialized to suppress compiler warning.
    int M = 0;
    switch (order_) {
      case StorageOrder::NHWC:
        channel_first = false;
        H = input.dim32(1);
        W = input.dim32(2);
        M = input.dim32(3);
        break;
      case StorageOrder::NCHW:
        channel_first = true;
        M = input.dim32(1);
        H = input.dim32(2);
        W = input.dim32(3);
        break;
      default:
        LOG(FATAL) << "Unknown Storage order: " << order_;
    }
    int output_height = 0, output_width = 0;
    ComputeSizeAndPad(
        H,
        stride_[0],
        kernel_[0],
        adj_[0],
        &pads_[0],
        &pads_[2],
        &output_height);
    ComputeSizeAndPad(
        W,
        stride_[1],
        kernel_[1],
        adj_[1],
        &pads_[1],
        &pads_[3],
        &output_width);
    if (channel_first) {
      output->Resize(N, output_channel, output_height, output_width);
    } else {
      output->Resize(N, output_height, output_width, output_channel);
    }
    VLOG(2) << "In: N " << N << " M " << M << " H " << H << " W " << W;
    VLOG(2) << "Out: output_channel " << output_channel << " H "
            << output_height << " W " << output_width;
  }

  bool RunOnDevice() override {
    switch (order_) {
      case StorageOrder::NHWC:
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        return RunOnDeviceWithOrderNCHW();
      default:
        LOG(FATAL) << "Unknown storage order: " << order_;
    }
    // To suppress old compiler warnings
    return true;
  }

  virtual bool RunOnDeviceWithOrderNCHW() {
    CAFFE_THROW("Not implemented");
  }

  virtual bool RunOnDeviceWithOrderNHWC() {
    CAFFE_THROW("Not implemented");
  }

  virtual ~ConvTransposeUnpoolBase() {}

 private:
  LegacyPadding legacy_pad_;
  int pad_;

 protected:
  vector<int> kernel_;
  vector<int> stride_;
  vector<int> pads_;
  vector<int> adj_;
  StorageOrder order_;
  bool shared_buffer_;
  Workspace* ws_;

  // Accessors for 2D conv params.

  inline int pad_t() const {
    return pads_[0];
  }

  inline int pad_l() const {
    return pads_[1];
  }

  inline int pad_b() const {
    return pads_[2];
  }

  inline int pad_r() const {
    return pads_[3];
  }

  inline int kernel_h() const {
    return kernel_[0];
  }

  inline int kernel_w() const {
    return kernel_[1];
  }

  inline int stride_h() const {
    return stride_[0];
  }

  inline int stride_w() const {
    return stride_[1];
  }

  inline int adj_h() const {
    return adj_[0];
  }

  inline int adj_w() const {
    return adj_[1];
  }

  inline void ComputeSizeAndPad(
      const int in_size,
      const int stride,
      const int kernel,
      const int adj,
      int* pad_head,
      int* pad_tail,
      int* out_size) {
    switch (legacy_pad_) {
      case LegacyPadding::NOTSET:
        CAFFE_ENFORCE(*pad_head >= 0);
        CAFFE_ENFORCE(*pad_tail >= 0);
        *out_size =
            (in_size - 1) * stride + kernel + adj - *pad_head - *pad_tail;
        break;
      // We handle cases of LegacyPadding::VALID and LegacyPadding::SAME
      // the same way
      case LegacyPadding::VALID:
      case LegacyPadding::SAME:
        *pad_head = 0;
        *pad_tail = 0;
        *out_size = (in_size - 1) * stride + kernel + adj;
        break;
      case LegacyPadding::CAFFE_LEGACY_POOLING:
        LOG(FATAL) << "CAFFE_LEGACY_POOLING is no longer supported.";
        break;
    }
  }
};

#define USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(Context) \
  USE_OPERATOR_FUNCTIONS(Context);                        \
  using ConvTransposeUnpoolBase<Context>::kernel_;        \
  using ConvTransposeUnpoolBase<Context>::stride_;        \
  using ConvTransposeUnpoolBase<Context>::pads_;          \
  using ConvTransposeUnpoolBase<Context>::adj_;           \
  using ConvTransposeUnpoolBase<Context>::order_;         \
  using ConvTransposeUnpoolBase<Context>::shared_buffer_; \
  using ConvTransposeUnpoolBase<Context>::ws_

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_TRANSPOSE_UNPOOL_OP_BASE_H_
