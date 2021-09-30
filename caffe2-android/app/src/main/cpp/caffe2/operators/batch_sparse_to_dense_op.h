// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_BATCH_SPARSE_TO_DENSE_OP_H_
#define CAFFE2_OPERATORS_BATCH_SPARSE_TO_DENSE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class BatchSparseToDenseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchSparseToDenseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(TIndex, "dense_last_dim", dense_last_dim_, -1),
        OP_SINGLE_ARG(T, "default_value", default_value_, static_cast<T>(0)) {}
  bool RunOnDevice() override;

  // TODO: enable the filler
  DISABLE_INPUT_FILLERS(Context)

 private:
  TIndex dense_last_dim_;
  T default_value_;
  INPUT_TAGS(LENGTHS, INDICES, VALUES);
};

template <typename T, class Context>
class BatchDenseToSparseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchDenseToSparseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  bool RunOnDevice() override;

 private:
  TIndex dense_last_dim_;
  INPUT_TAGS(LENGTHS, INDICES, DENSE);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_SPARSE_TO_DENSE_OP_H_
