/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class MarkerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound("Output(Out) should not be null"));
    auto in_dims = ctx->GetInputDim("X");
    std::string marker_role = ctx->Attrs().Get<std::string>("marker_role");
    std::string marker_pos = ctx->Attrs().Get<std::string>("marker_pos");

    VLOG(3) << "Marker oprator X.shape=" << in_dims << ";"
            << "The role is:" << marker_role << ";"
            << "The position is:" << marker_pos << ".";

    PADDLE_ENFORCE_NE(framework::product(in_dims), 0,
                      platform::errors::PreconditionNotMet(
                          "The Input variable X(%s) has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function.",
                          ctx->Inputs("X").front()));
  }
};

class MarkerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Input data (only used in CUDAKernel).");
    AddOutput("Out", "(Tensor) Output data (only used in CUDAKernel).");
    AddAttr<std::string>("marker_role","(string, default forward)forward or backward, mark different stages of porcess.")
                        .SetDefault("forward");
    AddAttr<std::string>("marker_pos","(string, default B)the posititon where the marker is placed, B stands for begin of duration, E stands for end of duration.")
                        .SetDefault("B");
    AddComment(R"DOC(
                      Marker Operator - Add marker at the beginning/end of a forward/backward process.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(marker, ops::MarkerOp, ops::MarkerOpMaker);