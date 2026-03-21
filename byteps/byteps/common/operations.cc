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

#include <cuda_runtime.h>
#include <unistd.h>
#include <torch/extension.h>
#include <cstring>
#include <memory>
#include <thread>
#include <string>
#include "compressor/compressor.h"
#include "compressor/compressor_registry.h"
#include "compressor/utils.h"
#include "core_loops.h"
#include "global.h"
#include "logging.h"
#include "operations.h"

namespace byteps {
namespace common {

extern "C" {

void byteps_init() {
  int use_gdr=std::stoi(getenv("DMLC_USE_GDR"));
  if(use_gdr){
    byteps_lazy_init_for_gdr();
  }else{
    byteps_lazy_init();
  }
  BytePSGlobal::GetOrInitPS();
}

void byteps_lazy_init_for_gdr(){
  BytePSGlobal::Init();
  std::vector<LoopFunction> func;

  // Push & Pull in distributed mode
  if (BytePSGlobal::IsDistributed()) {
    if (BytePSGlobal::IsRootDevice()) {
      for(int i=0;i<BytePSGlobal::GetPushThread();i++){
        func.push_back(PushLoopGDR);
      }
      for(int i=0;i<BytePSGlobal::GetPushThread();i++){
        func.push_back(PullLoopGDR);
      }
    }
  }
  BytePSGlobal::Start(func);
  return;
}

void byteps_lazy_init() {
  BytePSGlobal::Init();

  // The order of func does not matter
  std::vector<LoopFunction> func;

  // Push & Pull in distributed mode
  if (BytePSGlobal::IsDistributed()) {
    if (BytePSGlobal::IsRootDevice()) {
      for(int i=0;i<BytePSGlobal::GetPushThread();i++){
        func.push_back(PullLoop);
      }
      func.push_back(DecompressLoop);
    }
  }

  // Cross-PCIe-switch reduce
  if (BytePSGlobal::IsCrossPcieSwitch()) {
    func.push_back(PcieReduceLoop);
  }

  // Copy between GPU and CPU
  if (BytePSGlobal::IsCrossPcieSwitch() || BytePSGlobal::IsDistributed()) {
    func.push_back(CopyDevice2HostLoop);
    if (BytePSGlobal::IsRootDevice()) {
      // PUSH can be a real push in distributed mode
      // Or a dummy barrier in cross-pcie-switch mode
      for(int i=0;i<BytePSGlobal::GetPushThread();i++){
        func.push_back(PushLoop);
      }
      func.push_back(CompressLoop);
      func.push_back(RootCopyHost2DeviceLoop);
    } else {
      func.push_back(CoordinatePushLoop);
      func.push_back(NonRootCopyHost2DeviceLoop);
      func.push_back(NonRootCopyListenLoop);
    }
  }

  // Per-PCIe-switch NCCL calls
  func.push_back(SyncNcclLoop);
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    func.push_back(RootNcclLoop);
  } else {
    func.push_back(CoordinateReduceLoop);
    func.push_back(CoordinateBroadcastLoop);
    func.push_back(NonRootNcclLoop);
  }

  BytePSGlobal::Start(func);
  return;
}

void byteps_shutdown() {
  BytePSGlobal::Shutdown();
  BPS_LOG(DEBUG) << "BytePS has been completely shutdown now";
  return;
}

void byteps_resume(int num_workers, int num_servers) {
  // set ps, worker numbers
  BPS_LOG(DEBUG) << "Resume worker number: " << num_workers
                 << "DMLC_NUM_WORKER: " << getenv("DMLC_NUM_WORKER");
  BPS_LOG(DEBUG) << "Resume server number: " << num_workers
                 << "DMLC_NUM_SERVER: " << getenv("DMLC_NUM_SERVER");
  BPS_LOG(DEBUG) << "Start resuming BytePS";

  BytePSGlobal::SetResumingFlag(true);
  byteps_init();

  // redeclare tensor with original order
  BytePSGlobal::ReDeclareTensor();
  BytePSGlobal::SetResumingFlag(false);

  BPS_LOG(INFO) << "BytePS has been resumed now";
}

void byteps_suspend() {
  BPS_LOG(DEBUG) << "Start suspending BytePS";
  BytePSGlobal::Shutdown();
  BPS_LOG(INFO) << "BytePS has been suspended now";
  return;
}

int byteps_rank() { return BytePSGlobal::GetRank(); }

int byteps_local_rank() { return BytePSGlobal::GetLocalRank(); }

int byteps_size() { return BytePSGlobal::GetSize(); }

int byteps_local_size() { return BytePSGlobal::GetLocalSize(); }

}  // extern "C"

extern "C" PyObject* byteps_get_pushpull_speed() {
  auto entry = PushPullSpeed::GetSpeed();
  PyObject* ret = Py_BuildValue("(Kf)", entry->ts, entry->speed);

  return ret;
}

Status CheckInitialized() { return BytePSGlobal::CheckInit(); }

void PartitionTensor(
    std::shared_ptr<TensorTableEntry> entry,
    std::vector<std::shared_ptr<TensorTableEntry>> &partitions) {
  BPS_CHECK(entry->counter_ptr)
      << entry->tensor_name << " counter pointer is null";
  size_t size = entry->tensor ? entry->tensor->size() : entry->output->size();
  size_t bound = BytePSGlobal::GetPartitionBound();
  size_t accumulated = 0;
  int i = 0;

  while (accumulated < size) {
    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
    // will assign the key later, so don't do it now
    // e->key = entry->key;
    e->tensor_name = entry->tensor_name + std::string("_") + std::to_string(i);
    e->context = entry->context;
    e->ready_event = entry->ready_event;
    e->device = entry->device;
    e->priority = entry->priority;
    e->version = entry->version;
    e->callback = entry->callback;
    e->cpubuff = entry->cpubuff;
    e->gpu_ptr = entry->gpu_ptr;
    e->pcie_cpubuff = entry->pcie_cpubuff;
    e->queue_list = entry->queue_list;
    e->tensor = entry->tensor;
    e->output = entry->output;
    e->offset = accumulated;
    e->len = ((size - accumulated) > bound) ? bound : (size - accumulated);
    e->counter_ptr = entry->counter_ptr;
    e->total_partnum = entry->total_partnum;
    if (!entry->context->compressor_list.empty()) {
      e->compressor = entry->context->compressor_list[i];
    }

    accumulated += e->len;
    ++i;

    partitions.push_back(e);
  }
}

void PartitionTensorGDR(
    std::shared_ptr<TensorTableEntry> entry,
    std::vector<std::shared_ptr<TensorTableEntry>> &partitions) {
  
  BPS_CHECK(entry->counter_ptr)
      << entry->tensor_name << " counter pointer is null";

  // 1. 确定总大小 (以字节为单位)
  // 对于 GDR，必须精确知道字节大小以便计算指针偏移
  size_t size = entry->tensor ? entry->tensor->size() : entry->output->size();
  
  // 获取配置的分片大小 (BytePSGlobal::GetPartitionBound() 通常返回字节数)
  size_t bound = BytePSGlobal::GetPartitionBound();
  
  // GDR 优化：如果总大小小于分片界限，通常不需要分片，直接作为一个任务处理可能效率更高
  // 但为了保持逻辑一致性，这里保留循环，或者可以在外部加判断。
  // 注意：某些 RDMA 网卡对地址对齐有要求 (如 4KB 页对齐)，如果 bound 不是对齐的，可能需要调整。
  // 这里假设 bound 已经处理过对齐问题。

  size_t accumulated = 0;
  int i = 0;

  // 获取基地址指针 (Base Pointer)
  // 逻辑：如果设备是 GPU，优先使用 gpu_ptr；如果是 CPU 但使用了 Pin Memory 准备做 GDR，使用 cpubuff。
  // 注意：需要确保 entry->gpu_ptr 或 entry->tensor->data() 是有效的基地址。
  void* base_ptr = nullptr;
  bool is_gpu = (entry->device != CPU_DEVICE_ID);

  if (is_gpu) {
      // 优先使用显式存储的 gpu_ptr，如果为空则尝试从 tensor 获取
      if (entry->gpu_ptr != nullptr) {
          base_ptr = entry->gpu_ptr;
      } else if (entry->tensor) {
          base_ptr = const_cast<void*>(entry->tensor->data()); 
      } else if (entry->output) {
          base_ptr = const_cast<void*>(entry->output->data());
      }
  } else {
      // CPU 情况，通常使用 cpubuff (如果是 pinned memory)
      base_ptr = entry->cpubuff;
      if (base_ptr == nullptr && entry->tensor) {
           base_ptr = const_cast<void*>(entry->tensor->data());
      }
  }

  while (accumulated < size) {
    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
    
    // --- 基础属性拷贝 ---
    e->tensor_name = entry->tensor_name + std::string("_") + std::to_string(i);
    e->context = entry->context;
    e->ready_event = entry->ready_event;
    e->device = entry->device;
    e->priority = entry->priority;
    e->version = entry->version;
    e->callback = entry->callback;
    e->queue_list = entry->queue_list;
    e->counter_ptr = entry->counter_ptr;
    e->total_partnum = entry->total_partnum;
    
    // 共享原始 Tensor 对象 (浅拷贝)，实际数据访问通过 offset + base_ptr 进行
    e->tensor = entry->tensor;
    e->output = entry->output;

    // --- GDR 关键修改：计算分片后的指针和长度 ---
    
    // 计算当前分片的长度
    size_t remaining = size - accumulated;
    e->len = (remaining > bound) ? bound : remaining;
    e->offset = accumulated;

    // 计算当前分片的起始指针 (Base Ptr + Offset)
    if (base_ptr != nullptr) {
        uint8_t* byte_base = reinterpret_cast<uint8_t*>(base_ptr);
        void* chunk_ptr = byte_base + accumulated;
        
        if (is_gpu) {
            e->gpu_ptr = chunk_ptr; 
            // 对于 GDR，cpubuff 通常设为 nullptr 或者保持不变作为备用，
            // 关键在于确保下层通信库使用 gpu_ptr 进行注册和传输。
            e->cpubuff = nullptr; 
        } else {
            // CPU Pinned Memory for GDR
            e->cpubuff = chunk_ptr;
            e->gpu_ptr = nullptr;
        }
    } else {
        // 如果无法获取基地址，保持原样，依赖下层逻辑通过 offset 计算 (不推荐用于高性能 GDR)
        e->gpu_ptr = entry->gpu_ptr;
        e->cpubuff = entry->cpubuff;
    }

    // PCIe Merging buffers 也需要根据逻辑切分吗？
    // 通常 pcie_cpubuff 是一个向量，如果是多卡聚合场景，可能需要更复杂的逻辑。
    // 这里暂时深拷贝向量，具体地址偏移可能需要下层处理，或者如果它是用于中间缓冲，
    // 在纯 GDR 路径下可能根本用不到这个字段。
    // e->pcie_cpubuff = entry->pcie_cpubuff; 

    // --- 压缩器处理 ---
    // 注意：GDR 通常用于未压缩传输。如果开启了压缩，压缩器通常是针对整个 tensor 的，
    // 或者每个分片需要独立的压缩上下文。
    // 原逻辑：e->compressor = entry->context->compressor_list[i];
    // 如果 compressor_list 是按分片预分配好的，则保留；否则需小心。
    // if (!entry->context->compressor_list.empty()) {
    //     if (i < static_cast<int>(entry->context->compressor_list.size())) {
    //         e->compressor = entry->context->compressor_list[i];
    //     } else {
    //         // 防止越界，视具体压缩策略而定，可能需要复用或报错
    //         e->compressor = nullptr; 
    //     }
    // }

    // 初始化其他压缩相关字段
    e->compressed = nullptr; // 分片初始状态未压缩

    accumulated += e->len;
    ++i;

    partitions.push_back(e);
  }
}

Status EnqueueTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                     std::shared_ptr<Tensor> output,
                     std::shared_ptr<ReadyEvent> ready_event, const int device,
                     const int priority, const int version,
                     StatusCallback callback,
                     std::shared_ptr<std::vector<QueueType>> queue_list) {
  if (BytePSGlobal::ShouldShutdown()) {
    return Status::OK();
  }

  auto &name = context.tensor_name;
  if (input && output) {
    BPS_CHECK_EQ(input->size(), output->size())
        << name << " output tensor size does not match";
  }

  // add queue
  // if (BytePSGlobal::IsRootDevice() && !context.compressor_list.empty()) {
  //   auto it = std::find(queue_list->begin(), queue_list->end(), PUSH);
  //   // it = queue_list->insert(it, COMPRESS);  // before PUSH
  //   it = std::find(queue_list->begin(), queue_list->end(), PULL);
  //   // queue_list->insert(it + 1, DECOMPRESS);  // after PULL
  // }

  std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
  e->tensor_name = name;
  e->context = &context;
  e->tensor = input;
  e->output = output;
  e->ready_event = ready_event;
  e->device = device;
  e->priority = priority;
  e->version = version;
  e->callback = callback;

  if (device == CPU_DEVICE_ID) {
    cudaError_t err = cudaHostRegister(const_cast<void *>(input->data()),
                                       input->size(), cudaHostRegisterMapped);
    if (err == cudaSuccess) {
      BPS_LOG(DEBUG) << name
                     << " cpu address has changed, so it is pinned again.";
    }
    CUDA_CALL(cudaHostGetDevicePointer(&(context.gpu_ptr),
                                       const_cast<void *>(input->data()), 0));
  }

  e->cpubuff = context.cpubuff;
  e->gpu_ptr = context.gpu_ptr;
  e->pcie_cpubuff = context.pcie_cpubuff;
  e->queue_list = *queue_list;
  e->counter_ptr = std::make_shared<std::atomic_int>(0);
  e->total_partnum = context.key_list.size();

  std::vector<std::shared_ptr<TensorTableEntry>> partitions;
  PartitionTensor(e, partitions);
  BPS_CHECK_EQ(context.key_list.size(), partitions.size())
      << name << ": " << context.key_list.size() << ", " << partitions.size();

  if (e->queue_list.size() == 0) {
    BPS_CHECK(e->tensor_name != "");
    BPS_LOG(TRACE) << e->tensor_name << ", device=" << e->device
                   << " has no queue_list assigned, skipped";
    e->callback(Status::OK());
    return Status::OK();
  }

  // add for profiling
  if (context.profile_flag) {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    BPSCommTime *ret = new BPSCommTime;
    ret->start_t = (long long)(us.count());
    context.comm_time.push(ret);
  }

  size_t accumulated = 0;
  for (size_t i = 0; i < partitions.size(); ++i) {
    auto task = partitions[i];
    task->key = context.key_list[i];  // assign the key now
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "EnqueueTensor: " << (task->tensor_name)
                   << ", key=" << (task->key) << ", offset=" << (task->offset)
                   << ", len=" << (task->len) << ", device=" << (task->device)
                   << " rank=" << BytePSGlobal::GetLocalRank();

    BytePSGlobal::GetScheduledQueue(e->queue_list[0])->addTask(task);
    accumulated += task->len;
  }

  auto tensor = (e->tensor ? e->tensor : e->output);
  BPS_CHECK(tensor);
  BPS_CHECK_EQ(accumulated, tensor->size())
      << "accumulated partition size not equal to original tensor size";

  BPS_LOG(TRACE) << "EnqueueTensor finished: " << name
                 << ", rank=" << BytePSGlobal::GetLocalRank();
  return Status::OK();
}

// Status EnqueueTensorGDR(std::vector<torch::Tensor> tensor,
//                      std::shared_ptr<ReadyEvent> ready_event, const int device,
//                      const int priority, const int version,
//                      StatusCallback callback,
//                      std::shared_ptr<std::vector<QueueType>> queue_list) {
//   if (BytePSGlobal::ShouldShutdown()) {
//     return Status::OK();
//   }

//   // auto &name = context.tensor_name;
//   // if (input && output) {
//   //   BPS_CHECK_EQ(input->size(), output->size())
//   //       << name << " output tensor size does not match";
//   // }

//   // add queue
//   // if (BytePSGlobal::IsRootDevice() && !context.compressor_list.empty()) {
//     auto it = std::find(queue_list->begin(), queue_list->end(), PUSH);
//     // it = queue_list->insert(it, COMPRESS);  // before PUSH
//     it = std::find(queue_list->begin(), queue_list->end(), PULL);
//     // queue_list->insert(it + 1, DECOMPRESS);  // after PULL
//   // }

//   // std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
//   // e->tensor_name = name;
//   // e->context = &context;
//   // e->tensor = input;
//   // e->output = output;
//   // e->ready_event = ready_event;
//   // e->device = device;
//   // e->priority = priority;
//   // e->version = version;
//   // e->callback = callback;


//   // e->cpubuff = context.cpubuff;
//   // e->gpu_ptr = context.gpu_ptr;
//   // e->pcie_cpubuff = context.pcie_cpubuff;
//   // e->queue_list = *queue_list;
//   // e->counter_ptr = std::make_shared<std::atomic_int>(0);
//   // e->total_partnum = context.key_list.size();

//   std::vector<std::shared_ptr<TensorTableEntry>> partitions;
//   PartitionTensor(e, partitions);
//   BPS_CHECK_EQ(context.key_list.size(), partitions.size())
//       << name << ": " << context.key_list.size() << ", " << partitions.size();

//   if (e->queue_list.size() == 0) {
//     BPS_CHECK(e->tensor_name != "");
//     BPS_LOG(TRACE) << e->tensor_name << ", device=" << e->device
//                    << " has no queue_list assigned, skipped";
//     e->callback(Status::OK());
//     return Status::OK();
//   }

//   // add for profiling
//   if (context.profile_flag) {
//     auto now = std::chrono::system_clock::now();
//     auto duration = now.time_since_epoch();
//     auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

//     BPSCommTime *ret = new BPSCommTime;
//     ret->start_t = (long long)(us.count());
//     context.comm_time.push(ret);
//   }

//   size_t accumulated = 0;
//   for (size_t i = 0; i < partitions.size(); ++i) {
//     auto task = partitions[i];
//     task->key = context.key_list[i];  // assign the key now
//     BPS_CHECK(task->tensor_name != "");
//     BPS_LOG(TRACE) << "EnqueueTensor: " << (task->tensor_name)
//                    << ", key=" << (task->key) << ", offset=" << (task->offset)
//                    << ", len=" << (task->len) << ", device=" << (task->device)
//                    << " rank=" << BytePSGlobal::GetLocalRank();

//     BytePSGlobal::GetScheduledQueue(e->queue_list[0])->addTask(task);
//     accumulated += task->len;
//   }

//   auto tensor = (e->tensor ? e->tensor : e->output);
//   BPS_CHECK(tensor);
//   BPS_CHECK_EQ(accumulated, tensor->size())
//       << "accumulated partition size not equal to original tensor size";

//   BPS_LOG(TRACE) << "EnqueueTensor finished: " << name
//                  << ", rank=" << BytePSGlobal::GetLocalRank();
//   return Status::OK();
// }

void InitTensor(BPSContext &context, size_t size, int dtype, void *cpubuff) {
  std::lock_guard<std::mutex> lock(context.init_mutex);
  if (context.initialized) {
    return;
  }
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));

  BPS_CHECK_GT(size, 0) << "init tensor size not larger than 0";
  // Get metadata
  auto bound = BytePSGlobal::GetPartitionBound();
  auto &name = context.tensor_name;
  context.buff_len = size;
  size_t accumulated = 0;

  // Add for timeline
  BytePSGlobal::SetProfileFlag(&context);
  context.local_rank = BytePSGlobal::GetLocalRank();

  // Total key space is 0 to 2^64 - 1
  // It will be divided to N PS servers, for now we assume N <= 2^16
  // Then we have 2^48 key space left (top 16 bits for different servers)
  // MXNet server has a bug dealing with keys larger than 2^32
  // Below we support up to 2^16 tensors, and up to 2^16 partitions per tensor
  ps::Key start_key = context.declared_key << 16;
  while (accumulated < size) {
    context.key_list.push_back(start_key++);
    accumulated +=
        ((size - accumulated) > bound) ? bound : (size - accumulated);
  }
  BPS_LOG(DEBUG) << name << " partitioned to " << context.key_list.size()
                 << " part(s)"
                 << ", total_len=" << size << ", key_range=["
                 << context.key_list.front() << ", " << context.key_list.back()
                 << "]"
                 << " rank=" << BytePSGlobal::GetLocalRank();

  auto key_list = context.key_list;

  BPS_CHECK_GT(key_list.size(), 0) << name;
  BPS_CHECK_EQ(key_list.size(),
               (size_t)(size + bound - 1) / bound)  // round up
      << key_list.size() << ", size=" << size << ", bound=" << bound;

  BPS_LOG(TRACE) << "Begin init " << name << ", size=" << size
                 << ", parts=" << key_list.size();

  // If cpubuff is not nullptr, the tensor itself is on CPU
  // We need to register with CUDA so that NCCL can work on it
  if (cpubuff) {
    BPS_LOG(DEBUG) << name << " is already on cpu, len=" << size;
    cudaError_t e = cudaHostRegister(cpubuff, size, cudaHostRegisterMapped);
    if (e != cudaSuccess) {
      BPS_LOG(INFO) << cudaGetErrorString(e)
                    << " (You may ignore this if your program continues)";
    }
    CUDA_CALL(cudaHostGetDevicePointer(&(context.gpu_ptr), cpubuff, 0));
  }

  // We always allocate our own cpu buffer
  // use the first key in key_list as the index
  auto shm_obj = BytePSGlobal::GetSharedMemoryObj();

  size_t aligned_size = Align(size, dtype);
  if (BytePSGlobal::IsCrossPcieSwitch()) {
    context.pcie_cpubuff =
        shm_obj->openPcieSharedMemory(key_list[0], aligned_size);
    context.cpubuff = context.pcie_cpubuff.back();
  } else {
    context.cpubuff = shm_obj->openSharedMemory(std::string("BytePS_ShM_"),
                                                key_list[0], aligned_size);
  }
  BPS_LOG(TRACE) << name << ": open shared memory size " << aligned_size;

  // Init tensors with BytePS server
  char *data = const_cast<char *>(static_cast<const char *>(context.cpubuff));
  accumulated = 0;
  size_t i = 0;
  BPS_LOG(INFO) << "tensor size=" << size;
  // small tensor does not need to be compressed
  if (size < BytePSGlobal::GetMinCompressBound()) {
    context.kwargs.clear();
  }
  while (accumulated < size) {
    auto key = key_list[i];
    int len = ((size - accumulated) > bound) ? bound : (size - accumulated);

    if (BytePSGlobal::IsDistributed() && BytePSGlobal::IsRootDevice()) {
      auto ps = BytePSGlobal::GetOrInitPS();
      // encode the key for pskv scattering
      auto &pskv = BytePSGlobal::EncodeDefaultKey(key, len);
      // false means not to delete data when SArray is deleted
      ps::SArray<char> vals(data + accumulated, len, false);
      // cmd type
      int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
      // blocking push, also as a global barrirer
      ps->Wait(ps->ZPush(pskv.keys, vals, pskv.lens, cmd));

      // register
      if (!context.kwargs.empty()) {
        auto compressor_ptr = compressor::CompressorRegistry::Create(
            context.kwargs, Align(len, dtype), static_cast<DataType>(dtype));
        context.compressor_list.push_back(std::move(compressor_ptr));
      }
    }

    accumulated += len;
    ++i;
  }

  BPS_CHECK_EQ(accumulated, size);
  BPS_CHECK_EQ(i, key_list.size());

  // send to server
  if (!context.kwargs.empty() && BytePSGlobal::IsDistributed() &&
      BytePSGlobal::IsRootDevice()) {
    auto ps = BytePSGlobal::GetOrInitPS();
    auto content = compressor::Serialize(context.kwargs);
    auto len = content.size();
    auto data = const_cast<char *>(content.c_str());
    for (auto key : key_list) {
      auto &kv = BytePSGlobal::EncodeDefaultKey(key, len);
      ps::SArray<char> vals(data, len, false);
      int cmd = GetCommandType(RequestType::kCompressedPushPull, dtype);
      ps->Wait(ps->ZPush(kv.keys, vals, kv.lens, cmd));
    }
  }

  context.initialized = true;

  BPS_LOG(TRACE) << "Finish Init " << name << ", size=" << size
                 << ", parts=" << key_list.size();
}

BPSContext &GetContextFromName(const std::string &name) {
  return BytePSGlobal::GetContextFromName(name);
}

bool IsTensorDeclared(const std::string &name) {
  return BytePSGlobal::IsTensorDeclared(name);
}

void RegisterCompressor(const std::string &name,
                        std::unordered_map<std::string, std::string> &kwargs) {
  return BytePSGlobal::RegisterCompressor(name, kwargs);
}

std::shared_ptr<std::vector<QueueType>> GetPushQueueList(int device) {
  auto queue_list = std::make_shared<std::vector<QueueType>>();

  // Per-PCIe-switch NCCL reduce
  if(BytePSGlobal::IsUseGDR()){
    queue_list->push_back(PUSH);
    return queue_list;
  }

  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    queue_list->push_back(REDUCE);
  } else {
    queue_list->push_back(COORDINATE_REDUCE);
    queue_list->push_back(REDUCE);
  }

  // Copy from GPU to CPU
  if (BytePSGlobal::IsDistributed() || BytePSGlobal::IsCrossPcieSwitch()) {
    queue_list->push_back(COPYD2H);
  }

  // Cross-PCIe-switch reduce
  if (BytePSGlobal::IsCrossPcieSwitch()) {
    queue_list->push_back(PCIE_REDUCE);
  }

  // Push in distributed mode
  // In case IsCrossPcieSwitch(), PUSH runs as a dummy barrier
  if (BytePSGlobal::IsDistributed() || BytePSGlobal::IsCrossPcieSwitch()) {
    if (BytePSGlobal::IsRootDevice()) {
      queue_list->push_back(PUSH);
    } else {
      queue_list->push_back(COORDINATE_PUSH);
    }
  }
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetPullQueueList(int device) {
  auto queue_list = std::make_shared<std::vector<QueueType>>();

  if(BytePSGlobal::IsUseGDR()){
    queue_list->push_back(PULL);
    return queue_list;
  }
  // Pull in distributed mode
  if (BytePSGlobal::IsDistributed()) {
    if (BytePSGlobal::IsRootDevice()) {
      queue_list->push_back(PULL);
    }
  }

  // Copy from CPU to GPU
  if (BytePSGlobal::IsDistributed() || BytePSGlobal::IsCrossPcieSwitch()) {
    queue_list->push_back(COPYH2D);
  }

  // Per-PCIe-switch NCCL broadcast
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    queue_list->push_back(BROADCAST);
  } else {
    queue_list->push_back(COORDINATE_BROADCAST);
    queue_list->push_back(BROADCAST);
  }
  return queue_list;
}

}  // namespace common
}  // namespace byteps
