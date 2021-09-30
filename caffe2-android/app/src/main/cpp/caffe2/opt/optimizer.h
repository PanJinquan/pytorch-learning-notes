#ifndef CAFFE2_OPT_OPTIMIZER_H
#define CAFFE2_OPT_OPTIMIZER_H

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {
namespace opt {

NetDef optimize(NetDef net, Workspace* ws, int level = 1);
NetDef optimize(NetDef net, int level = 1);

} // namespace opt
} // namespace caffe2

#endif // CAFFE2_OPT_OPTIMIZER_H
