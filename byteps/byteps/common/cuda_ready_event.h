// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef BYTEPS_CUDA_READY_EVENT_H
#define BYTEPS_CUDA_READY_EVENT_H

#include <memory>

#include "common.h"

namespace byteps {
namespace common {

class CudaReadyEvent : public ReadyEvent {
 public:
  explicit CudaReadyEvent(cudaStream_t stream);
  ~CudaReadyEvent() override;
  bool Ready() const override;

 private:
  int device_ = CPU_DEVICE_ID;
  cudaEvent_t event_ = nullptr;
};

std::shared_ptr<ReadyEvent> RecordReadyEventOnStream(cudaStream_t stream);

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_CUDA_READY_EVENT_H
