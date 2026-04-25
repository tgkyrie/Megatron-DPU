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

#include "core_loops.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "common.h"
#include "compressor/compressor.h"
#include "global.h"
#include "logging.h"

namespace byteps {
namespace common {

namespace {

inline bool IsRdmaMrDebugEnabled() {
  static bool enabled = []() {
    const char* v = getenv("BYTEPS_RDMA_MR_DEBUG");
    return v && atoi(v) != 0;
  }();
  return enabled;
}

inline bool IsGdrLenProbeEnabled() {
  static bool enabled = []() {
    const char* v = getenv("BYTEPS_GDR_LEN_PROBE");
    return v && atoi(v) != 0;
  }();
  return enabled;
}

inline bool IsGdrCpuFallbackEnabled() {
  static bool enabled = []() {
    const char* v = getenv("BYTEPS_GDR_FALLBACK");
    return v ? atoi(v) != 0 : true;
  }();
  return enabled;
}

inline size_t GdrCpuFallbackThresholdBytes() {
  static size_t threshold = []() {
    constexpr size_t kMiB = 1024UL * 1024UL;
    const char* explicit_mb = getenv("BYTEPS_GDR_FALLBACK_THRESHOLD_MB");
    if (explicit_mb) {
      long mb = atol(explicit_mb);
      return mb > 0 ? static_cast<size_t>(mb) * kMiB : 0;
    }

    const char* cache_mb_env = getenv("BYTEPS_RDMA_MR_CACHE_LIMIT_MB");
    long cache_mb = cache_mb_env ? atol(cache_mb_env) : 192;
    if (cache_mb > 0) {
      return static_cast<size_t>(cache_mb) * kMiB * 3 / 4;
    }
    return 128UL * kMiB;
  }();
  return threshold;
}

inline bool ShouldUseGdrCpuFallback(
    const std::shared_ptr<TensorTableEntry>& task, size_t tensor_size) {
  if (!IsGdrCpuFallbackEnabled()) return false;
  if (!task || task->device == CPU_DEVICE_ID || task->cpubuff == nullptr) {
    return false;
  }
  size_t threshold = GdrCpuFallbackThresholdBytes();
  return tensor_size >= threshold;
}

inline size_t ClampLenByCudaAddressRange(const char* tag, uint64_t key, char* ptr,
                                         size_t len) {
  if (!ptr || len == 0) return len;

  cudaPointerAttributes attr;
  auto st = cudaPointerGetAttributes(&attr, ptr);
  if (st != cudaSuccess) {
    if (IsRdmaMrDebugEnabled()) {
      fprintf(stderr,
              "[GDR-DEBUG-ADDR] %s key=%llu cudaPointerGetAttributes failed: "
              "ptr=%p len=%zu err=%s\n",
              tag, (unsigned long long)key, ptr, len, cudaGetErrorString(st));
      fflush(stderr);
    }
    return len;
  }

#if CUDART_VERSION >= 10000
  auto mem_type = attr.type;
#else
  auto mem_type = attr.memoryType;
#endif

  if (IsRdmaMrDebugEnabled()) {
    fprintf(stderr,
            "[GDR-DEBUG-ADDR] %s key=%llu ptr=%p mem_type=%d device=%d "
            "device_ptr=%p host_ptr=%p len=%zu\n",
            tag, (unsigned long long)key, ptr, (int)mem_type, attr.device,
#if CUDART_VERSION >= 10000
            attr.devicePointer, attr.hostPointer,
#else
            attr.devicePointer, attr.hostPointer,
#endif
            len);
    fflush(stderr);
  }

  if (mem_type != cudaMemoryTypeDevice) {
    if (IsRdmaMrDebugEnabled()) {
      fprintf(stderr,
              "[GDR-DEBUG-ADDR] %s key=%llu ptr=%p is not device memory "
              "(mem_type=%d)\n",
              tag, (unsigned long long)key, ptr, (int)mem_type);
      fflush(stderr);
    }
  }

  return len;
}

inline bool ProbeDeviceReadableByte(const char* p) {
  if (!p) return false;
  char v = 0;
  auto st = cudaMemcpy(&v, p, 1, cudaMemcpyDeviceToHost);
  if (st == cudaSuccess) return true;
  // clear sticky error status so subsequent CUDA calls are not polluted
  cudaGetLastError();
  return false;
}

inline size_t ClampLenByCudaReadableProbe(const char* tag, uint64_t key, char* ptr,
                                          size_t len) {
  if (!ptr || len == 0) return len;

  // Fast path: tail byte readable => keep original len.
  if (ProbeDeviceReadableByte(ptr + (len - 1))) {
    if (IsRdmaMrDebugEnabled()) {
      fprintf(stderr,
              "[GDR-DEBUG-LEN] %s key=%llu probe keep len=%zu (ptr=%p)\n", tag,
              (unsigned long long)key, len, ptr);
      fflush(stderr);
    }
    return len;
  }

  // If first byte is already unreadable, this pointer is not usable.
  if (!ProbeDeviceReadableByte(ptr)) {
    fprintf(stderr,
            "[GDR-DEBUG-LEN] %s key=%llu unreadable base ptr=%p, len=%zu\n",
            tag, (unsigned long long)key, ptr, len);
    fflush(stderr);
    return 0;
  }

  // Binary search the maximum readable prefix length.
  size_t lo = 1;      // readable
  size_t hi = len;    // not readable at hi-1
  while (lo + 1 < hi) {
    size_t mid = lo + ((hi - lo) >> 1);
    if (ProbeDeviceReadableByte(ptr + (mid - 1))) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  fprintf(stderr,
          "[GDR-DEBUG-LEN] %s key=%llu clip by cuda probe: len %zu -> %zu "
          "(ptr=%p)\n",
          tag, (unsigned long long)key, len, lo, ptr);
  fflush(stderr);
  return lo;
}

}  // namespace

void FinishOrProceed(std::shared_ptr<TensorTableEntry> task) {
  auto& queue_list = task->queue_list;
  BPS_CHECK_GE(queue_list.size(), 1);
  auto this_op = queue_list[0];
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->reportFinish(task->len);
  if (BytePSGlobal::IsTensorSampled(task->key)) {
    // We only support sampling
    BPS_CHECK(task->tensor->dtype() == common::BYTEPS_FLOAT32);
    size_t i = task->offset / 4;
    size_t j = (task->offset + task->len) / 4 - 1;
    if (task->device == CPU_DEVICE_ID) {
      BPS_LOG(DEBUG) << "Sampled key=" << task->key
                     << " rank=" << BytePSGlobal::GetLocalRank()
                     << " input[0]=" << *((float*)(task->tensor->data()) + i)
                     << "\tinput[-1]=" << *((float*)(task->tensor->data()) + j)
                     << "\toutput[0]=" << *((float*)(task->output->data()) + i)
                     << "\toutput[-1]=" << *((float*)(task->output->data()) + j)
                     << "\t after stage: " << LogStrings[this_op];
    } else {
      float i0, i1, o0, o1;
      cudaMemcpy(&i0, (float*)(task->tensor->data()) + i, 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&i1, (float*)(task->tensor->data()) + j, 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&o0, (float*)(task->output->data()) + i, 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&o1, (float*)(task->output->data()) + j, 4,
                 cudaMemcpyDeviceToHost);
      BPS_LOG(DEBUG) << "Sampled key=" << task->key
                     << " rank=" << BytePSGlobal::GetLocalRank()
                     << " input[0]=" << i0 << "\tinput[-1]=" << i1
                     << "\toutput[0]=" << o0 << "\toutput[-1]=" << o1
                     << "\t after stage: " << LogStrings[this_op];
    }
  }

  if (task->context->profile_flag) {
    BPS_CHECK(task->context->part_comm_time[task->key][this_op].back()->dur ==
              0)
        << " tensor: " << task->tensor_name << " task->key:" << task->key
        << " type:" << this_op << " 'dur' has already been assigned:"
        << task->context->part_comm_time[task->key][this_op].back()->dur;
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    auto _ts =
        task->context->part_comm_time[task->key][this_op].back()->start_t;
    BPS_CHECK(task->context->part_comm_time.find(task->key) !=
              task->context->part_comm_time.end())
        << " tensor: " << task->tensor_name << " task->key:" << task->key
        << " type:" << this_op;
    BPS_CHECK(task->context->part_comm_time[task->key].find(this_op) !=
              task->context->part_comm_time[task->key].end())
        << " tensor: " << task->tensor_name << " task->key:" << task->key
        << " type:" << this_op;

    task->context->part_comm_time[task->key][this_op].back()->dur =
        (long long)(us.count()) - _ts;
  }

  // finish current QueueType of this task, erase current QueueType.
  queue_list.erase(queue_list.begin());
  if (queue_list.size() > 0) {
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank() << " finishes "
                   << LogStrings[this_op] << ", tensor: " << task->tensor_name
                   << ", key=" << task->key << "; Passing to the next queue.";
    BytePSGlobal::GetScheduledQueue(queue_list[0])->addTask(task);
  } else {
    // this is the last QueueType of this current sub-task.
    BPS_CHECK(task->counter_ptr) << task->tensor_name << " counter_ptr is null";
    int v = task->counter_ptr.get()->fetch_add(1);
    if (v == (int)(task->total_partnum - 1)) {
      // if meet this condition, that means all sub-tasks of this task have been
      // done
      BPS_CHECK(task->tensor_name != "");
      BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank()
                     << " finish processing tensor: " << task->tensor_name;

      if (PushPullSpeed::ShouldRecord()) {
        PushPullSpeed::RecordSpeed(task);
      }

      task->callback(Status::OK());
      //* Add for profiling communication events
      if (task->context->profile_flag) {
        BPS_CHECK(task->context->comm_time.back()->dur == 0)
            << " tensor: " << task->tensor_name
            << " 'dur' has already been assigned:"
            << task->context->comm_time.back()->dur;
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(duration);
        auto _ts = task->context->comm_time.back()->start_t;
        task->context->comm_time.back()->dur = (long long)(us.count()) - _ts;
      }
      // Set the profile_flag first
      // *step_cnt* denotes the number this gradient has been synchronized.
      task->context->step_cnt += 1;
      BytePSGlobal::SetProfileFlag(task->context);
    }
  }
  return;
}

bool RunCoordinateLoopOnce(QueueType this_op) {
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    int rank = BytePSGlobal::GetLocalRank();
    auto key = task->key;

    // first send to next queue and then broadcast signal
    // to guarantee the entry is available when getTask(key) at Reduce/Broadcast
    // thread
    FinishOrProceed(task);

    BytePSCommSignal sig = PUSH_READY;
    std::shared_ptr<BytePSComm> comm;

    switch (this_op) {
      case COORDINATE_REDUCE: {
        sig = REDUCE_READY;
        comm = BytePSGlobal::GetNccl()->GetSignalComm();
        break;
      }
      case COORDINATE_BROADCAST: {
        sig = BCAST_READY;
        comm = BytePSGlobal::GetNccl()->GetSignalComm();
        break;
      }
      case COORDINATE_PUSH: {
        sig = PUSH_READY;
        comm = BytePSGlobal::GetBasicComm();
        break;
      }
      default:
        BPS_CHECK(0) << "unsupported op: " << this_op;
    }

    BPS_CHECK_NE(rank, comm->getRoot())
        << "only non-root device should enter COORDINATE loop";

    struct BytePSCommMsg msg = {rank, sig, key};
    comm->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << task->tensor_name << " send coordinate info: "
                   << "Signal=" << sig << ", rank=" << rank << ", key=" << key;

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

inline void PostNcclCalls(
    std::shared_ptr<byteps::common::TensorTableEntry> task, QueueType this_op) {
  BPS_CHECK(this_op == REDUCE || this_op == BROADCAST)
      << "Only REDUCE and BROADCAST use NCCL.";
  auto tensor = (this_op == REDUCE) ? task->tensor : task->output;
  BPS_CHECK(tensor);
  BPS_CHECK_EQ(0, tensor->size() % tensor->shape().num_elements());

  auto key = task->key;
  auto len = task->len;
  auto offset = task->offset;
  auto unit_len = tensor->size() / tensor->shape().num_elements();
  auto p = (char*)(tensor->data()) + offset;
  if (task->device == CPU_DEVICE_ID) {
    p = (char*)(task->gpu_ptr) + offset;
  }

  auto nccl_dtype = getNcclDataType(tensor->dtype());

  auto nccl = BytePSGlobal::GetNccl();
  auto nccl_stream = nccl->GetStream(key, this_op);
  auto nccl_comm = nccl->GetComm(key, this_op);
  auto nccl_root = nccl->GetRoot(key, this_op);
  auto nccl_size = nccl->GetSize();
  auto nccl_rank = nccl->GetRank(key, this_op);

  auto num_elem_per_gpu = len / nccl_size / unit_len;
  auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);
  if (BytePSGlobal::IsUsingReduce()) {
    nccl_root = BytePSGlobal::GetReduceRootByKey(key);
    num_elem_per_gpu = 0;
    left_elem = len / unit_len;
    BPS_LOG(TRACE) << "Reduce key=" << key << " to root=" << nccl_root
                   << " rank=" << BytePSGlobal::GetLocalRank();
  }

  BPS_CHECK(task->tensor_name != "");
  BPS_LOG(TRACE) << task->tensor_name << " calling NCCL " << LogStrings[this_op]
                 << " (rank=" << nccl_rank << ") key=" << key
                 << ", elements=" << len / unit_len
                 << ", device=" << task->device;

  if (this_op == REDUCE) {
    // We reduce to task->output except that it is a CPU tensor
    auto out_p = (char*)(task->output->data()) + offset;
    if (task->device == CPU_DEVICE_ID && task->tensor == task->output) {
      out_p = p;
    }

    if (num_elem_per_gpu) {
      NCCLCHECK(ncclReduceScatter(
          (const void*)p,
          (void*)(out_p + nccl_rank * num_elem_per_gpu * unit_len),
          (size_t)num_elem_per_gpu, (ncclDataType_t)nccl_dtype,
          (ncclRedOp_t)ncclSum, (ncclComm_t)nccl_comm,
          (cudaStream_t)nccl_stream));
    }
    if (left_elem) {
      NCCLCHECK(ncclReduce((const void*)(p + len - left_elem * unit_len),
                           (void*)(out_p + len - left_elem * unit_len),
                           (size_t)left_elem, (ncclDataType_t)nccl_dtype,
                           (ncclRedOp_t)ncclSum, (int)nccl_root,
                           (ncclComm_t)nccl_comm, (cudaStream_t)nccl_stream));
    }
  } else {
    if (num_elem_per_gpu) {
      NCCLCHECK(ncclAllGather(
          (const void*)(p + nccl_rank * num_elem_per_gpu * unit_len), (void*)p,
          (size_t)num_elem_per_gpu, (ncclDataType_t)nccl_dtype,
          (ncclComm_t)nccl_comm, (cudaStream_t)nccl_stream));
    }
    if (left_elem) {
      NCCLCHECK(ncclBroadcast((const void*)(p + len - left_elem * unit_len),
                              (void*)(p + len - left_elem * unit_len),
                              (size_t)left_elem, (ncclDataType_t)nccl_dtype,
                              (int)nccl_root, (ncclComm_t)nccl_comm,
                              (cudaStream_t)nccl_stream));
    }
  }
}

bool RunRootNcclLoopOnce() {
  auto signal_comm = BytePSGlobal::GetNccl()->GetSignalComm();
  int root = signal_comm->getRoot();
  int rank = BytePSGlobal::GetLocalRank();
  BPS_CHECK_EQ(rank, root);

  int nccl_size = BytePSGlobal::GetNccl()->GetSize();
  QueueType nccl_ops[] = {REDUCE, BROADCAST};

  auto nccl_entry = std::make_shared<NcclGroupEntry>();
  auto& tasks = nccl_entry->tasks;
  auto& queues = nccl_entry->queues;

  NCCLCHECK(ncclGroupStart());
  for (auto this_op : nccl_ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    for (int i = 0; i < BytePSGlobal::GetNccl()->GetGroupSize(); i++) {
      auto task = q->getTask();
      if (!task) {
        break;
      }
      tasks.push_back(task);
      queues.push_back(q);

      if (nccl_size > 1) {
        // notify non-root devices
        struct BytePSCommMsg msg = {
            rank, (this_op == REDUCE) ? DO_REDUCE : DO_BROADCAST, task->key};
        signal_comm->broadcastSignal(&msg, sizeof(BytePSCommMsg));
        PostNcclCalls(task, this_op);
      }
    }
  }
  if (tasks.size()) {
    struct BytePSCommMsg msg = {rank, DO_GROUP, 0};
    signal_comm->broadcastSignal(&msg, sizeof(BytePSCommMsg));
    NCCLCHECK(ncclGroupEnd());
    nccl_entry->RecordEvents();
    BPS_LOG(TRACE) << "NCCL Group size=" << tasks.size() << " rank=" << rank;
    BytePSGlobal::GetNccl()->EnqueueGroup(nccl_entry);
  } else {
    NCCLCHECK(ncclGroupEnd());
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunNonRootNcclLoopOnce() {
  auto signal_comm = BytePSGlobal::GetNccl()->GetSignalComm();
  int root = signal_comm->getRoot();
  int rank = BytePSGlobal::GetLocalRank();
  BPS_CHECK_NE(rank, root);

  auto nccl_entry = std::make_shared<NcclGroupEntry>();
  auto& tasks = nccl_entry->tasks;
  auto& queues = nccl_entry->queues;
  struct BytePSCommMsg msg = {};

  NCCLCHECK(ncclGroupStart());
  while (1) {
    signal_comm->recvSignalFromRoot(&msg, sizeof(BytePSCommMsg));
    if (BytePSGlobal::ShouldShutdown()) return true;
    if (msg.signal == DO_GROUP) {
      break;
    }
    QueueType this_op = REDUCE;
    if (msg.signal == DO_BROADCAST) {
      this_op = BROADCAST;
    } else {
      BPS_CHECK_EQ(msg.signal, DO_REDUCE) << msg.signal << ", " << DO_REDUCE;
    }

    auto key = msg.key;

    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask(key);
    BPS_CHECK(task);

    tasks.push_back(task);
    queues.push_back(q);

    PostNcclCalls(task, this_op);
  }
  NCCLCHECK(ncclGroupEnd());

  nccl_entry->RecordEvents();
  BytePSGlobal::GetNccl()->EnqueueGroup(nccl_entry);
  return true;
}

bool RunSyncNcclOnce() {
  auto nccl_entry = BytePSGlobal::GetNccl()->DequeueGroup();
  if (nccl_entry) {
    nccl_entry->SynchronizeEvents();
    for (size_t i = 0; i < nccl_entry->tasks.size(); i++) {
      FinishOrProceed(nccl_entry->tasks[i]);
    }
    nccl_entry->DestroyEvents();
    BPS_LOG(TRACE) << "Finished NCCL Group size=" << nccl_entry->tasks.size()
                   << " rank=" << BytePSGlobal::GetLocalRank();
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunCopyDevice2HostLoopOnce() {
  QueueType this_op = COPYD2H;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();

  if (task) {
    auto copy_d2h_Stream = BytePSGlobal::GetCopyDevice2HostStream();
    // If we ran NCCL reduce, we should copy from task->output
    auto tensor =
        (BytePSGlobal::GetNccl()->GetSize() > 1) ? task->output : task->tensor;
    BPS_CHECK(tensor);
    auto key = task->key;

    auto nccl = BytePSGlobal::GetNccl();
    auto nccl_root = nccl->GetRoot(key, REDUCE);
    auto nccl_size = nccl->GetSize();
    auto nccl_rank = nccl->GetRank(key, REDUCE);

    auto len = task->len;
    auto offset = task->offset;
    auto p = (char*)(tensor->data()) + offset;
    if (task->device == CPU_DEVICE_ID) {
      p = (char*)(task->gpu_ptr) + offset;
    }
    auto unit_len = tensor->size() / tensor->shape().num_elements();
    char* cpubuff;
    if (BytePSGlobal::IsCrossPcieSwitch()) {
      BPS_CHECK(task->pcie_cpubuff.size());
      cpubuff =
          (char*)(task->pcie_cpubuff[BytePSGlobal::GetPcieSwitchIndex()]) +
          offset;
    } else {
      cpubuff = (char*)(task->cpubuff) + offset;
    }

    BPS_CHECK(cpubuff) << task->tensor_name
                       << ": CPU buffer not initialized, size=" << len;

    auto num_elem_per_gpu = len / nccl_size / unit_len;
    auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

    auto copy_offset = nccl_rank * num_elem_per_gpu * unit_len;
    auto copy_len = num_elem_per_gpu * unit_len;
    if (left_elem && (nccl_root == nccl_rank)) {
      copy_len += left_elem * unit_len;
    }

    if (BytePSGlobal::IsUsingReduce()) {
      copy_offset = 0;
      copy_len = (BytePSGlobal::GetReduceRootByKey(key) == nccl_rank) ? len : 0;
    }

    if (copy_len) {
      CUDA_CALL(cudaMemcpyAsync(
          (void*)(cpubuff + copy_offset), (const void*)(p + copy_offset),
          (size_t)copy_len, (cudaMemcpyKind)cudaMemcpyDeviceToHost,
          (cudaStream_t)*copy_d2h_Stream));
      CUDA_CALL(cudaStreamSynchronize(*copy_d2h_Stream));
    }

    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunPcieReduceLoopOnce() {
  BPS_CHECK(BytePSGlobal::IsCrossPcieSwitch());
  QueueType this_op = PCIE_REDUCE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    auto reducer = BytePSGlobal::GetCpuReducer();
    if (!reducer->isRoot()) {
      // send signal to root
      int rank = BytePSGlobal::GetLocalRank();
      auto key = task->key;
      BytePSCommSignal sig = PCIE_REDUCE_READY;
      struct BytePSCommMsg msg = {rank, sig, key};
      reducer->getComm()->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));
    } else {
      auto tensor = task->tensor;

      auto key = task->key;
      auto len = task->len;
      auto offset = task->offset;
      auto unit_len = tensor->size() / tensor->shape().num_elements();

      auto nccl = BytePSGlobal::GetNccl();
      auto nccl_root = nccl->GetRoot(key, REDUCE);
      auto nccl_size = nccl->GetSize();
      auto nccl_rank = nccl->GetRank(key, REDUCE);

      auto num_elem_per_gpu = len / nccl_size / unit_len;
      auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

      auto copy_len = num_elem_per_gpu * unit_len;
      if (left_elem && (nccl_root == nccl_rank)) {
        copy_len += left_elem * unit_len;
      }

      if (copy_len) {
        auto total_offset = offset + nccl_rank * num_elem_per_gpu * unit_len;

        // Below we assume there are only two PCIe switch
        // and we run reducer in the context of the second switch
        reducer->sum((void*)((char*)(task->cpubuff) + total_offset),
                     (void*)((char*)(task->pcie_cpubuff[0]) + total_offset),
                     copy_len, tensor->dtype());
      }
    }

    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunCompressLoopOnce() {
  QueueType this_op = COMPRESS;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter COMPRESS loop";
    BPS_CHECK(task->compressor != nullptr);
    BPS_CHECK(task->compressed == nullptr);

    // spawn
    BytePSGlobal::GetThreadPool()->enqueue([task]() {
      char* data = const_cast<char*>(static_cast<const char*>(task->cpubuff) +
                                     task->offset);
      int len = task->len;
      int dtype = task->tensor->dtype();
      compressor::tensor_t grad(data, len, dtype);
      auto compressed = task->compressor->Compress(grad);
      BPS_CHECK_LE(compressed.size, len)
          << "Compressor Implementation Error "
          << ", key=" << task->key << ", src_len=" << len
          << ", compressed_len=" << compressed.size;

      task->compressed = std::make_shared<decltype(compressed)>(compressed);

      // restore rt
      auto& queue_list = task->queue_list;
      BytePSGlobal::GetScheduledQueue(queue_list[1])
          ->reset(task->key, BytePSGlobal::GetLocalSize() - 1);

      FinishOrProceed(task);
    });

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunPushLoopOnce() {
  QueueType this_op = PUSH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter PUSH loop";

    if (BytePSGlobal::IsDistributed()) {
      auto offset = task->offset;
      auto len = task->len;

      char* data;
      BPS_CHECK(task->cpubuff);
      data =
          const_cast<char*>(static_cast<const char*>(task->cpubuff) + offset);

      // get metadata
      const int dtype = task->tensor->dtype();

      // use compressed data/len
      if (task->compressed) {
        BPS_LOG(DEBUG) << "PUSH with gradient compression. key=" << task->key;
        data = task->compressed->data;
        len = task->compressed->size;
        task->compressed = nullptr;
      }

      // false means not to delete data when SArray is deleted
      ps::SArray<char> vals(data, len, false);

      int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
      auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
      BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd,
                                   [task, q]() { FinishOrProceed(task); });
    } else {
      // This is a dummy barrier for IsCrossPcieSwitch()
      BPS_CHECK(BytePSGlobal::IsCrossPcieSwitch());
      FinishOrProceed(task);
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunGDRPushLoopOnce() {
  QueueType this_op = PUSH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter PUSH loop";
    auto tensor = task->tensor;
    if (BytePSGlobal::IsDistributed()) {
      bool gdr_debug = IsRdmaMrDebugEnabled();
      size_t offset = static_cast<size_t>(task->offset);
      size_t len = static_cast<size_t>(task->len);
      auto orig_len = len;
      auto tensor_size = tensor ? tensor->size() : 0;

      if (tensor_size > 0) {
        if (offset >= tensor_size) {
          if (gdr_debug) {
            fprintf(stderr,
                    "[GDR-DEBUG-LEN] PUSH key=%llu invalid offset=%zu "
                    "tensor_size=%zu\n",
                    (unsigned long long)task->key, (size_t)offset,
                    (size_t)tensor_size);
            fflush(stderr);
          }
          BPS_CHECK_LT(offset, tensor_size);
        }
        if (len > tensor_size - offset) {
          auto clipped = tensor_size - offset;
          if (gdr_debug) {
            fprintf(stderr,
                    "[GDR-DEBUG-LEN] PUSH key=%llu clip len from %zu to %zu "
                    "(offset=%zu tensor_size=%zu)\n",
                    (unsigned long long)task->key, (size_t)len,
                    (size_t)clipped, (size_t)offset, (size_t)tensor_size);
            fflush(stderr);
          }
          len = clipped;
        }
      }

      if (gdr_debug) {
        fprintf(stderr,
                "[GDR-DEBUG-PUSH] key=%llu task->device=%d task->gpu_ptr=%p "
                "task->cpubuff=%p tensor->data()=%p offset=%lu len=%zu "
                "tensor_size=%zu\n",
                (unsigned long long)task->key, task->device, task->gpu_ptr,
                task->cpubuff, tensor->data(), (unsigned long)offset, len,
                (size_t)tensor_size);
        fflush(stderr);
      }

      char* data;
      bool gdr_cpu_fallback = ShouldUseGdrCpuFallback(task, tensor_size);
      if (gdr_cpu_fallback) {
        CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
        const void* gpu_base = task->gpu_ptr ? task->gpu_ptr : tensor->data();
        auto* gpu_data =
            const_cast<char*>(static_cast<const char*>(gpu_base) + offset);
        data = const_cast<char*>(
            static_cast<const char*>(task->cpubuff) + offset);
        CUDA_CALL(cudaMemcpy(data, gpu_data, len, cudaMemcpyDeviceToHost));
        if (gdr_debug) {
          fprintf(stderr,
                  "[GDR-DEBUG-PUSH] CPU fallback key=%llu gpu=%p cpu=%p "
                  "len=%zu tensor_size=%zu threshold=%zu\n",
                  (unsigned long long)task->key, gpu_data, data, len,
                  (size_t)tensor_size, GdrCpuFallbackThresholdBytes());
          fflush(stderr);
        }
      } else if (task->device == CPU_DEVICE_ID) {
        const void* base = task->cpubuff ? task->cpubuff : tensor->data();
        data = const_cast<char*>(static_cast<const char*>(base) + offset);
        if (gdr_debug) {
          fprintf(stderr,
                  "[GDR-DEBUG-PUSH] CPU device: base=%p (%s), data=%p\n",
                  base, task->cpubuff ? "cpubuff" : "tensor->data()", data);
          fflush(stderr);
        }
      } else {
        const void* base = task->gpu_ptr ? task->gpu_ptr : tensor->data();
        data = const_cast<char*>(static_cast<const char*>(base) + offset);
        if (gdr_debug) {
          fprintf(stderr,
                  "[GDR-DEBUG-PUSH] GPU device: base=%p (%s), data=%p\n",
                  base, task->gpu_ptr ? "gpu_ptr" : "tensor->data()", data);
          fflush(stderr);
        }
      }

      if (gdr_debug) {
        fprintf(stderr, "[GDR-DEBUG-PUSH] Final data address: %p\n",
                (void*)data);
        fflush(stderr);
      }
      if (task->device != CPU_DEVICE_ID && !gdr_cpu_fallback) {
        len = ClampLenByCudaAddressRange("PUSH", task->key, data, len);
        if (IsGdrLenProbeEnabled()) {
          len = ClampLenByCudaReadableProbe("PUSH", task->key, data, len);
        }
        BPS_CHECK_GT(len, 0) << "PUSH key=" << task->key
                             << " clamped length to zero for ptr=" << (void*)data;
      }

      // get metadata
      const int dtype = task->tensor->dtype();
      int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
      auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
      if (pskv.size > 0 && pskv.size != len) {
        if (gdr_debug) {
          fprintf(stderr,
                  "[GDR-DEBUG-LEN] PUSH key=%llu adjust len from %zu to %zu\n",
                  (unsigned long long)task->key, (size_t)orig_len,
                  (size_t)pskv.size);
          fflush(stderr);
        }
        len = pskv.size;
      }

      // false means not to delete data when SArray is deleted
      ps::SArray<char> vals(data, len, false);

      BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd,
                                   [task, q]() { FinishOrProceed(task); });
    } else {
      // This is a dummy barrier for IsCrossPcieSwitch()
      BPS_CHECK(BytePSGlobal::IsCrossPcieSwitch());
      FinishOrProceed(task);
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunGDRPullLoopOnce() {
  QueueType this_op = PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter PULL loop";
    auto tensor = task->tensor;
    auto output = task->output;
    // TODO: allow merging
    bool gdr_debug = IsRdmaMrDebugEnabled();
    size_t offset = static_cast<size_t>(task->offset);
    size_t len = static_cast<size_t>(task->len);
    auto orig_len = len;
    auto tensor_size = output ? output->size() : (tensor ? tensor->size() : 0);

    if (tensor_size > 0) {
      if (offset >= tensor_size) {
        if (gdr_debug) {
          fprintf(stderr,
                  "[GDR-DEBUG-LEN] PULL key=%llu invalid offset=%zu "
                  "tensor_size=%zu\n",
                  (unsigned long long)task->key, (size_t)offset,
                  (size_t)tensor_size);
          fflush(stderr);
        }
        BPS_CHECK_LT(offset, tensor_size);
      }
      if (len > tensor_size - offset) {
        auto clipped = tensor_size - offset;
        if (gdr_debug) {
          fprintf(stderr,
                  "[GDR-DEBUG-LEN] PULL key=%llu clip len from %zu to %zu "
                  "(offset=%zu tensor_size=%zu)\n",
                  (unsigned long long)task->key, (size_t)len,
                  (size_t)clipped, (size_t)offset, (size_t)tensor_size);
          fflush(stderr);
        }
        len = clipped;
      }
    }

    if (gdr_debug) {
      fprintf(stderr,
              "[GDR-DEBUG-PULL] key=%llu task->device=%d task->gpu_ptr=%p "
              "task->cpubuff=%p tensor->data()=%p output->data()=%p "
              "offset=%lu len=%zu tensor_size=%zu\n",
              (unsigned long long)task->key, task->device, task->gpu_ptr,
              task->cpubuff, tensor->data(), output ? output->data() : nullptr,
              (unsigned long)offset, len, (size_t)tensor_size);
      fflush(stderr);
    }

    char* data;
    char* fallback_gpu_data = nullptr;
    bool gdr_cpu_fallback = ShouldUseGdrCpuFallback(task, tensor_size);
    if (gdr_cpu_fallback) {
      const void* gpu_base = output
                                 ? output->data()
                                 : (task->gpu_ptr ? task->gpu_ptr
                                                  : tensor->data());
      fallback_gpu_data =
          const_cast<char*>(static_cast<const char*>(gpu_base) + offset);
      data =
          const_cast<char*>(static_cast<const char*>(task->cpubuff) + offset);
      if (gdr_debug) {
        fprintf(stderr,
                "[GDR-DEBUG-PULL] CPU fallback key=%llu gpu=%p cpu=%p "
                "len=%zu tensor_size=%zu threshold=%zu\n",
                (unsigned long long)task->key, fallback_gpu_data, data, len,
                (size_t)tensor_size, GdrCpuFallbackThresholdBytes());
        fflush(stderr);
      }
    } else if (task->device == CPU_DEVICE_ID) {
      const void* base = task->cpubuff
                             ? task->cpubuff
                             : (output ? output->data() : tensor->data());
      data = const_cast<char*>(static_cast<const char*>(base) + offset);
      if (gdr_debug) {
        fprintf(stderr, "[GDR-DEBUG-PULL] CPU device: base=%p, data=%p\n",
                base, data);
        fflush(stderr);
      }
    } else {
      const void* base = output
                             ? output->data()
                             : (task->gpu_ptr ? task->gpu_ptr : tensor->data());
      data = const_cast<char*>(static_cast<const char*>(base) + offset);
      if (gdr_debug) {
        fprintf(stderr, "[GDR-DEBUG-PULL] GPU device: base=%p, data=%p\n",
                base, data);
        fflush(stderr);
      }
    }

    if (gdr_debug) {
      fprintf(stderr, "[GDR-DEBUG-PULL] Final data address: %p\n", (void*)data);
      fflush(stderr);
    }
    if (task->device != CPU_DEVICE_ID && !gdr_cpu_fallback) {
      len = ClampLenByCudaAddressRange("PULL", task->key, data, len);
      if (IsGdrLenProbeEnabled()) {
        len = ClampLenByCudaReadableProbe("PULL", task->key, data, len);
      }
      BPS_CHECK_GT(len, 0) << "PULL key=" << task->key
                           << " clamped length to zero for ptr=" << (void*)data;
    }

    // get metadata
    const int dtype = output ? output->dtype() : tensor->dtype();
    int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
    auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
    if (pskv.size > 0 && pskv.size != len) {
      if (gdr_debug) {
        fprintf(stderr,
                "[GDR-DEBUG-LEN] PULL key=%llu adjust len from %zu to %zu\n",
                (unsigned long long)task->key, (size_t)orig_len,
                (size_t)pskv.size);
        fflush(stderr);
      }
      len = pskv.size;
    }

    // false means not to delete data when SArray is deleted
    auto vals = new ps::SArray<char>(data, len, false);

    // issue pull
    if (gdr_cpu_fallback) {
      char* cpu_data = data;
      size_t copy_len = len;
      BytePSGlobal::GetPS()->ZPull(
          pskv.keys, vals, &pskv.lens, cmd,
          [vals, task, q, fallback_gpu_data, cpu_data, copy_len]() {
            CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
            CUDA_CALL(cudaMemcpy(fallback_gpu_data, cpu_data, copy_len,
                                 cudaMemcpyHostToDevice));
            delete vals;
            FinishOrProceed(task);
          });
    } else {
      BytePSGlobal::GetPS()->ZPull(pskv.keys, vals, &pskv.lens, cmd,
                                   [vals, task, q]() {
                                     delete vals;
                                     FinishOrProceed(task);
                                   });
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunPullLoopOnce() {
  QueueType this_op = PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter PULL loop";
    // TODO: allow merging
    auto offset = task->offset;
    auto len = task->len;

    char* data;
    BPS_CHECK(task->cpubuff);
    data = const_cast<char*>(static_cast<const char*>(task->cpubuff) + offset);

    // get metadata
    const int dtype = task->output->dtype();

    // false means not to delete data when SArray is deleted
    auto vals = new ps::SArray<char>(data, len, false);

    int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
    auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
    // issue pull
    BytePSGlobal::GetPS()->ZPull(pskv.keys, vals, &pskv.lens, cmd,
                                 [vals, task, q]() {
                                   delete vals;
                                   FinishOrProceed(task);
                                 });
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunDecompressLoopOnce() {
  QueueType this_op = DECOMPRESS;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter DECOMPRESS loop";
    BPS_CHECK(task->compressor != nullptr);

    // spawn
    BytePSGlobal::GetThreadPool()->enqueue([task]() {
      char* data = const_cast<char*>(static_cast<const char*>(task->cpubuff) +
                                     task->offset);
      auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, 0);
      auto len = pskv.lens[0];
      int dtype = task->tensor->dtype();
      compressor::tensor_t compressed(data, len, dtype);
      auto decompressed = task->compressor->Decompress(compressed);
      BPS_LOG(DEBUG) << "PULL with gradient compression. key=" << task->key;

      FinishOrProceed(task);
    });

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void CopyHost2Device(std::shared_ptr<byteps::common::TensorTableEntry> task) {
  auto copy_h2d_stream = BytePSGlobal::GetCopyHost2DeviceStream();
  auto tensor = task->output;
  BPS_CHECK(tensor);
  auto key = task->key;
  auto nccl = BytePSGlobal::GetNccl();
  auto nccl_root = nccl->GetRoot(key, BROADCAST);
  auto nccl_size = nccl->GetSize();
  auto nccl_rank = nccl->GetRank(key, BROADCAST);
  auto len = task->len;
  auto offset = task->offset;
  auto cpubuff = (char*)(task->cpubuff) + offset;
  BPS_CHECK(cpubuff) << task->tensor_name
                     << ": CPU buffer not initialized, size=" << len;

  auto gpu_addr = (char*)(tensor->data()) + offset;
  if (task->device == CPU_DEVICE_ID) {
    gpu_addr = (char*)(task->gpu_ptr) + offset;
  }

  auto unit_len = tensor->size() / tensor->shape().num_elements();
  auto num_elem_per_gpu = len / nccl_size / unit_len;
  auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

  auto copy_offset = nccl_rank * num_elem_per_gpu * unit_len;
  auto copy_len = num_elem_per_gpu * unit_len;
  if (left_elem && (nccl_root == nccl_rank)) {
    copy_len += left_elem * unit_len;
  }

  if (BytePSGlobal::IsUsingReduce()) {
    copy_offset = 0;
    copy_len = (BytePSGlobal::GetReduceRootByKey(key) == nccl_rank) ? len : 0;
  }

  if (copy_len) {
    CUDA_CALL(cudaMemcpyAsync(
        (void*)(gpu_addr + copy_offset), (const void*)(cpubuff + copy_offset),
        (size_t)copy_len, (cudaMemcpyKind)cudaMemcpyHostToDevice,
        (cudaStream_t)*copy_h2d_stream));
    CUDA_CALL(cudaStreamSynchronize(*copy_h2d_stream));
  }

  return;
}

bool RunRootCopyHost2DeviceLoopOnce() {
  QueueType this_op = COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();

  if (task) {
    auto key = task->key;
    int local_rank = BytePSGlobal::GetLocalRank();
    int local_size = BytePSGlobal::GetLocalSize();

    if (local_size > 1) {
      // notify non-root devices
      struct BytePSCommMsg msg = {local_rank, DO_COPYH2D, key};
      BytePSGlobal::GetBasicComm()->broadcastSignal(&msg,
                                                    sizeof(BytePSCommMsg));
    }
    CopyHost2Device(task);

    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunNonRootCopyListenLoopOnce() {
  auto signal_comm = BytePSGlobal::GetBasicComm();
  int root = signal_comm->getRoot();
  int rank = BytePSGlobal::GetLocalRank();
  BPS_CHECK_NE(root, rank);

  struct BytePSCommMsg msg = {};

  signal_comm->recvSignalFromRoot(&msg, sizeof(BytePSCommMsg));
  if (BytePSGlobal::ShouldShutdown()) return true;
  BPS_CHECK_EQ(msg.signal, DO_COPYH2D) << msg.signal;

  BytePSGlobal::GetCopyTable()->AddReadyCount(msg.key);

  BPS_LOG(TRACE) << "NonRootCopyListenLoop recved from root"
                 << ", signal=" << msg.signal << ", key=" << msg.key
                 << ", myrank=" << rank;
  return true;
}

bool RunNonRootCopyHost2DeviceLoopOnce() {
  QueueType this_op = COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();

  if (task) {
    CopyHost2Device(task);
    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void CoordinateReduceLoop() {
  while (RunCoordinateLoopOnce(COORDINATE_REDUCE) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CoordinateBroadcastLoop() {
  while (RunCoordinateLoopOnce(COORDINATE_BROADCAST) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CoordinatePushLoop() {
  while (RunCoordinateLoopOnce(COORDINATE_PUSH) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void PcieReduceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunPcieReduceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void RootNcclLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootNcclLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunNonRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void SyncNcclLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunSyncNcclOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CopyDevice2HostLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunCopyDevice2HostLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CompressLoop() {
  while (RunCompressLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void PushLoop() {
  while (RunPushLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void PushLoopGDR() {
  while (RunGDRPushLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void PullLoopGDR() {
  while (RunGDRPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void PullLoop() {
  while (RunPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void DecompressLoop() {
  while (RunDecompressLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void RootCopyHost2DeviceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunRootCopyHost2DeviceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyListenLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunNonRootCopyListenLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyHost2DeviceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
  while (RunNonRootCopyHost2DeviceLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

}  // namespace common
}  // namespace byteps
