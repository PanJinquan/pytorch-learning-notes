#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {
namespace opt {

void OptimizeForIdeep(
    nom::repr::NNModule* nn,
    caffe2::Workspace* ws,
    bool training_mode = false);
}
} // namespace caffe2
