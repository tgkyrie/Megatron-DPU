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

#include "cuda_ready_event.h"

#include <mutex>
#include <queue>
#include <unordered_map>

#include "logging.h"

namespace byteps {
namespace common {
namespace {

struct CudaReadyEventRegistry {
  std::unordered_map<int, std::queue<cudaEvent_t> > events;
  std::mutex mutex;
};

CudaReadyEventRegistry registry;

}  // namespace

CudaReadyEvent::CudaReadyEvent(cudaStream_t stream) {
  CUDA_CALL(cudaGetDevice(&device_));
  {
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto& queue = registry.events[device_];
    if (!queue.empty()) {
      event_ = queue.front();
      queue.pop();
    } else {
      CUDA_CALL(cudaEventCreateWithFlags(
          &event_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
  }
  CUDA_CALL(cudaEventRecord(event_, stream));
}

CudaReadyEvent::~CudaReadyEvent() {
  if (!event_) return;
  std::lock_guard<std::mutex> guard(registry.mutex);
  registry.events[device_].push(event_);
}

bool CudaReadyEvent::Ready() const {
  auto status = cudaEventQuery(event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  CUDA_CALL(status);
  return true;
}

std::shared_ptr<ReadyEvent> RecordReadyEventOnStream(cudaStream_t stream) {
  return std::make_shared<CudaReadyEvent>(stream);
}

}  // namespace common
}  // namespace byteps
