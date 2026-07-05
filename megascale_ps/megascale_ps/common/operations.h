// Copyright 2019 Bytedance Inc. All Rights Reserved.
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

#ifndef MEGASCALE_PS_OPERATIONS_H
#define MEGASCALE_PS_OPERATIONS_H

#include <functional>
#include "common.h"

namespace megascale_ps {
namespace common {

// Check that megascale_ps is initialized.
Status CheckInitialized();

extern "C" {

// C interface to initialize megascale_ps.
void megascale_ps_init();

// C interface to initialize megascale_ps (without initializing ps-lite).
void megascale_ps_lazy_init();

void megascale_ps_lazy_init_for_gdr();

// C interface to shut down megascale_ps.
void megascale_ps_shutdown();

// C interface to restart megascale_ps.
void megascale_ps_resume(int num_workers, int num_servers);

// C interface to suspend megascale_ps.
void megascale_ps_suspend();

// C interface to get index of current megascale_ps process.
// Returns -1 if megascale_ps is not initialized.
int megascale_ps_rank();

// C interface to get index of current megascale_ps process in the node it is on.
// Returns -1 if megascale_ps is not initialized.
int megascale_ps_local_rank();

// C interface to return number of megascale_ps processes.
// Returns -1 if megascale_ps is not initialized.
int megascale_ps_size();

// C interface to return number of megascale_ps processes in the node it is on.
// Returns -1 if megascale_ps is not initialized.
int megascale_ps_local_size();
}

extern "C" PyObject* megascale_ps_get_pushpull_speed();

// Below are all for Framework plugins
Status EnqueueTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                     std::shared_ptr<Tensor> output,
                     std::shared_ptr<ReadyEvent> ready_event, const int device,
                     const int priority, const int version,
                     StatusCallback callback,
                     std::shared_ptr<std::vector<QueueType>> queue_list);

void InitTensor(BPSContext &context, size_t size, int dtype, void *cpubuff);

// Only call these in Framework plugins for the best performance
bool IsTensorDeclared(const std::string &name);

void RegisterTensorExpectedWorkers(const std::string& name, int expected_workers);

void RegisterCompressor(const std::string &name,
                        std::unordered_map<std::string, std::string> &kwargs);

BPSContext &GetContextFromName(const std::string &name);

std::shared_ptr<std::vector<QueueType>> GetPushQueueList(int device);

std::shared_ptr<std::vector<QueueType>> GetPullQueueList(int device);

}  // namespace common
}  // namespace megascale_ps

#endif  // MEGASCALE_PS_OPERATIONS_H
