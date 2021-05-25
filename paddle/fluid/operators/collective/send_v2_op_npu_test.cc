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

#ifndef _WIN32
#include <unistd.h>
#endif

#include <stdio.h>
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "gtest/gtest.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

#include "paddle/fluid/operators/collective/gen_hccl_id_op_helper.h"
#include "paddle/fluid/operators/collective/send_v2_op.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(send_v2);
USE_NO_KERNEL_OP(c_gen_hccl_id);
USE_NO_KERNEL_OP(c_comm_init_hccl);
USE_OP_DEVICE_KERNEL(send_v2, NPU);

void PrepareUniqueId(f::Scope* scope, const p::DeviceContext& ctx,
                     HcclRootInfo* hccl_id) {
  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));
  int peer_id = atoi(getenv("PEER_ID"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; peer_id = " << peer_id;

  std::vector<int> rank_ids{0,1};
  f::AttributeMap gen_hccl_id;

  std::vector<std::string> endpointList = {"127.0.0.1:6175", "127.0.0.1:6177","127.0.0.1:6179", "127.0.0.1:6181","127.0.0.1:6183", "127.0.0.1:6185","127.0.0.1:6187", "127.0.0.1:6189"};
  gen_hccl_id["rank"] = rank_id;
  gen_hccl_id["endpoint"] = endpointList[device_id];
  std::vector<std::string> other_endpoints = {
      endpointList[peer_id]};
  gen_hccl_id["other_endpoints"] = other_endpoints;

  auto out = scope->Var("Out");
  auto id = out->GetMutable<HcclRootInfo>();

  VLOG(3) << "break";

  auto comm_init_op = f::OpRegistry::CreateOp("c_gen_hccl_id", {},
                                              {{"Out", {"Out"}}}, gen_hccl_id);
  VLOG(3) << "break";
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();

  memcpy(hccl_id, id, 1024);
}

void Prepare(f::Scope* scope, const p::DeviceContext& ctx,
             HcclRootInfo* hccl_id) {
  auto x = scope->Var("X");
  auto id = x->GetMutable<HcclRootInfo>();

  memcpy(id, hccl_id, 1024);

  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));
  int peer_id = atoi(getenv("PEER_ID"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; peer_id = " << peer_id;

  // std::vector<int> rank_ids{0, 1};
  f::AttributeMap comm_init_attrs;
  comm_init_attrs["ring_id"] = 0;
  comm_init_attrs["rank_ids"] = 2;
  comm_init_attrs["rank"] = rank_id;
  comm_init_attrs["device_id"] = device_id;
  // comm_init_attrs["rank_ids"] = rank_ids;
  auto comm_init_op = f::OpRegistry::CreateOp(
      "c_comm_init_hccl", {{"X", {"X"}}}, {}, comm_init_attrs);
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();
}

void TestHcomSendOp(f::Scope* scope, const p::DeviceContext& ctx) {
  std::cout << "BEGIN TEST:" << __FUNCTION__ << std::endl;
  auto x = scope->Var("Data");
  auto tensor_x = x->GetMutable<f::LoDTensor>();
  int num = atoi(getenv("DATA_SIZE"));

  EXPECT_GT(num, 0);
  EXPECT_LT(num, 1 << 15);
  std::vector<float> init(num * num, 1.0 * atoi(getenv("PEER_ID")));
  int rank_id = atoi(getenv("RANK_ID"));
  VLOG(3) << "rank id:" << rank_id;
  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({num, num});
  ctx.Wait();
  auto place = ctx.GetPlace();
  ctx.Wait();

  f::AttributeMap attrs;
  attrs["tag"] = std::string("srtest");
  attrs["peer"] = 0;
  attrs["ring_id"] = 0;
  attrs["srTag"] = 0;

  auto op = f::OpRegistry::CreateOp("send_v2", {{"X", {"Data"}}}, {}, attrs);

  for (int i = 0; i < 10; i++) {
    op->Run(*scope, place);
  }
  VLOG(3) << "send run over";
  ctx.Wait();
}

TEST(send_v2, NPU) {
  f::Scope scope;
  HcclRootInfo hccl_id;

  char* npu_id = getenv("FLAGS_selected_npus");
  VLOG(3) << "Select npu:" << npu_id;
  p::NPUDeviceContext ctx(p::NPUPlace(atoi(npu_id)));

  PrepareUniqueId(&scope, ctx, &hccl_id);
  Prepare(&scope, ctx, &hccl_id);
  TestHcomSendOp(&scope, ctx);
}
