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

#include "server.h"

#include <cstring>

#include "../common/compressor/utils.h"
#include "queue.h"

namespace byteps {
namespace server {

using namespace ps;

struct LatencyLogger {
  struct Event {
    std::string name;
    int64_t ns;
  };

  std::chrono::high_resolution_clock::time_point start;
  std::vector<Event> events;
  std::mutex mutex_;

  LatencyLogger() : start(std::chrono::high_resolution_clock::now()) {}

  void record_event(const std::string& name) {
    if (!record_event_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::high_resolution_clock::now();
    int64_t t =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - start)
            .count();
    events.push_back({name, t});
  }

  ~LatencyLogger() {
    if (events.empty()) return;
    std::ostringstream oss;
    oss << "Latency events:\n";
    for (const auto& e : events) {
      oss << "  [" << e.name << "] " << e.ns / 1000 << " us\n";
    }
    LOG(INFO) << oss.str();
  }
};

LatencyLogger logger_;
// engine related
std::vector<PriorityQueue*> engine_queues_;
std::vector<std::thread*> engine_threads_;

std::shared_ptr<KeyState> GetKeyState(uint64_t key) {
  std::lock_guard<std::mutex> lock(key_states_mu_);
  auto it = key_states_.find(key);
  if (it != key_states_.end()) {
    return it->second;
  }
  auto state = std::make_shared<KeyState>();
  key_states_[key] = state;
  return state;
}

void RegisterExpectedWorkers(KeyState* state, uint64_t key, int expected_workers) {
  if (state->expected_workers == -1) {
    state->expected_workers = expected_workers;
    return;
  }
  CHECK_EQ(state->expected_workers, expected_workers)
      << "Key " << key
      << " registered with inconsistent expected_workers: "
      << state->expected_workers
      << " vs " << expected_workers;
}

int GetExpectedWorkers(const KeyState& state) {
  if (state.expected_workers == -1) {
    return ps::NumWorkers();
  }
  return state.expected_workers;
}

void SendPushResponse(KeyState* state, uint64_t key, const ps::KVMeta& req,
                      ps::KVServer<char>* server) {
  logger_.record_event("push resp begin.key." + std::to_string(key));
  if (!state->has_push_response) {
    state->push_response = ps::KVPairs<char>();
    state->has_push_response = true;
  }
  server->Response(req, state->push_response);
  logger_.record_event("push resp end.key." + std::to_string(key));
}

void SendPullResponse(KeyState* state, const DataHandleType type,
                      const uint64_t key,
                      const ps::KVMeta& req_meta, ps::KVServer<char>* server) {
  logger_.record_event("pull resp begin.key." + std::to_string(key));
  char* data;
  size_t len;
  if (sync_mode_) {
    CHECK(state->update.merged.tensor) << "init " << key << " first";
    data = state->update.merged.tensor;
    len = state->update.merged.len;
  } else {
    CHECK(state->store.tensor) << "init " << key << " first";
    data = state->store.tensor;
    len = state->store.len;
  }

  // send pull response
  if (!state->has_pull_response) {
    state->pull_response.keys = {EncodeKey(key)};
    state->pull_response.lens = {len};
    state->pull_response.vals =
        ps::SArray<char>(data, len, false);  // zero copy
    state->has_pull_response = true;
    server->Response(req_meta, state->pull_response);
  } else {  // not new key, then reuse the memory address to avoid ibv_reg_mr on
            // RDMA data path
    auto p = static_cast<char*>(data);
    CHECK(p);
    state->pull_response.lens = {len};
    state->pull_response.vals = ps::SArray<char>(p, len, false);
    server->Response(req_meta, state->pull_response);
  }
  logger_.record_event("pull resp end.key." + std::to_string(key));
}

void BytePSServerEngineThread(int i) {
  auto& q = engine_queues_[i];
  while (true) {
    BytePSEngineMessage msg;
    q->WaitAndPop(&msg);
    if (msg.ops == TERMINATE) break;
    // do some check
    CHECK(msg.dst);
    CHECK(msg.src);
    CHECK(msg.state);
    auto state = msg.state;
    logger_.record_event("engine thread preprocess begin.");
    common::compressor::Compressor* compressor = nullptr;
    {
      std::lock_guard<std::mutex> lock(state->mu);
      compressor = state->compressor.get();
    }
    if (compressor) {
      // compress
      if (msg.ops == ALL_RECV) {
        common::compressor::tensor_t grad(reinterpret_cast<char*>(msg.src),
                                          msg.len, msg.type.dtype);
        auto compressed = compressor->Compress(grad);
        // 1. compress
        std::lock_guard<std::mutex> lock(state->mu);
        state->update.merged.tensor = compressed.data;
        state->update.merged.len = compressed.size;
      } else {  // decompress
        auto compressed_len = msg.sarray.lens[0];
        CHECK_LE(compressed_len, msg.len);
        common::compressor::tensor_t compressed(
            reinterpret_cast<char*>(msg.src), compressed_len, msg.type.dtype);
        auto decompressed = compressor->Decompress(compressed);
        msg.src = decompressed.data;
      }
    } else {
      if (msg.ops == ALL_RECV) {
        // 2. no compress
        std::lock_guard<std::mutex> lock(state->mu);
        state->update.merged.tensor = reinterpret_cast<char*>(msg.src);
        state->update.merged.len = msg.len;
      }
    }
    logger_.record_event("engine thread preprocess end.");

    bool is_debug = (debug_mode_ && (debug_key_ == msg.key));
    switch (msg.ops) {
      case COPY_FIRST: {
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_BEFORE \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
        logger_.record_event("copy data begin");
        bps_reducer_->copy(msg.dst, msg.src, msg.len);
        logger_.record_event("copy data end");
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_AFTER \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
      } break;

      case ALL_RECV: {
        std::lock_guard<std::mutex> lock(state->mu);
        state->is_push_finished = true;

        auto it = state->q_pull_reqmeta.begin();
        while (it != state->q_pull_reqmeta.end()) {
          if (state->seen_sender.find(it->sender) == state->seen_sender.end()) {
            SendPullResponse(state.get(), msg.type, msg.key, *it, byteps_server_);
            state->pull_cnt += 1;
            state->seen_sender.insert(it->sender);
            it = state->q_pull_reqmeta.erase(it);
          } else {
            ++it;
          }
          if (state->pull_cnt == (size_t)GetExpectedWorkers(*state)) {
            state->is_push_finished = false;
            state->pull_cnt = 0;
            state->seen_sender.clear();
            break;
          }
        }
      } break;

      case SUM_RECV: {
        auto bps_type = bps_reducer_->GetDataType(msg.type.dtype);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_BEFORE \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
        logger_.record_event("reduce begin");
        CHECK_GE(bps_reducer_->sum(msg.dst, msg.src, msg.len, bps_type), 0);
        logger_.record_event("reduce end");
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_AFTER \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
      } break;
      default:
        CHECK(0);
    }
  }
}  // namespace server

inline uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

void BytePSHandler(const ps::KVMeta& req_meta,
                   const ps::KVPairs<char>& req_data,
                   ps::KVServer<char>* server) {
  logger_.record_event("handler begin");
  std::unique_lock<std::mutex> global_lock(handle_mu_, std::defer_lock);
  if (use_global_handler_lock_) {
    global_lock.lock();
  }
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  // CHECK_EQ(type.requestType, RequestType::kDefaultPushPull);
  // do some check
  CHECK_EQ(req_data.keys.size(), (size_t)1);
  if (log_key_info_) {
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
      LOG(INFO) << "push key=" << DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender
                << "\t size=" << (size_t)req_data.lens[0];
    } else {
      LOG(INFO) << "pull key=" << (uint64_t)DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender;
    }
  }
  uint64_t key = DecodeKey(req_data.keys[0]);
  auto state = GetKeyState(key);
  std::lock_guard<std::mutex> state_lock(state->mu);

  if (type.requestType == RequestType::kGroupRegister) {
    CHECK(req_meta.push) << "group registration must be a push request";
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.lens[0], (int)sizeof(int));
    CHECK_EQ(req_data.vals.size(), (size_t)sizeof(int));
    int expected_workers = 0;
    std::memcpy(&expected_workers, req_data.vals.data(), sizeof(int));
    CHECK_GT(expected_workers, 0) << "invalid expected_workers for key=" << key;
    RegisterExpectedWorkers(state.get(), key, expected_workers);
    SendPushResponse(state.get(), key, req_meta, server);
    return;
  }

  // register compressor
  if (type.requestType == RequestType::kCompressedPushPull) {
    if (!state->compressor) {
      std::string content{reinterpret_cast<char*>(req_data.vals.data()),
                          static_cast<size_t>(req_data.lens[0])};
      auto kwargs = byteps::common::compressor::Deserialize(content);
      auto& stored = state->store;
      size_t aligned_size = byteps::common::Align(stored.len, stored.dtype);
      auto compressor_ptr =
          byteps::common::compressor::CompressorRegistry::Create(
              kwargs, aligned_size,
              static_cast<byteps::common::DataType>(stored.dtype));
      CHECK_NE(compressor_ptr, nullptr);
      state->compressor = std::move(compressor_ptr);
      if (log_key_info_) {
        LOG(INFO) << "register compressor for key=" << key;
      }
    }

    // buffer the request meta
    auto& updates = state->update;
    updates.request.push_back(req_meta);
    // should send response after collecting all init push
    if (updates.request.size() < (size_t)GetExpectedWorkers(*state)) return;

    for (const auto& req : updates.request) {
      SendPushResponse(state.get(), key, req, server);
    }
    updates.request.clear();
    return;
  }

  if (req_meta.push) {  // push request
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    auto& stored = state->store;
    auto len = (size_t)req_data.lens[0];
    auto recved = reinterpret_cast<char*>(req_data.vals.data());

    if (!stored.tensor) {
      auto& updates = state->update;
      if (sync_mode_) {
        updates.merged.len = len;
        updates.merged.dtype = type.dtype;
      }
      // buffer the request meta
      updates.request.push_back(req_meta);
      // should send response after collecting all init push
      if (updates.request.size() < (size_t)GetExpectedWorkers(*state)) return;
      // init stored buffer, use page aligned memory
      size_t aligned_size = common::Align(len, type.dtype);
      PageAlignedMalloc((void**)&stored.tensor, aligned_size);
      stored.len = len;
      stored.dtype = type.dtype;
      CHECK(stored.tensor);

      memset(stored.tensor, 0, stored.len);
      //   bps_reducer_->copy(stored->tensor, recved,
      //                      len);  // we may not need this copy
      for (const auto& req : updates.request) {
        SendPushResponse(state.get(), key, req, server);
      }
      updates.request.clear();
    } else {  // not first iteration
      auto& updates = state->update;
      auto tid = GetThreadID(key, len);
      if (updates.request.empty()) {  // from the first incoming worker
        if (sync_mode_) {
          if (debug_mode_ && (debug_key_ == key)) {
            std::lock_guard<std::mutex> lock(debug_mu_);
            LOG(INFO) << "stage: FIRST_WORKER_RECV \t"
                      << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored.tensor)
                      << "\t"
                      << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                      << "len: " << len << "\t"
                      << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
          }
          updates.merged.tmp_sarray = req_data;
          // copy
          // first worker packet, copy
          BytePSEngineMessage msg = {
              timestamp_.fetch_add(1, std::memory_order_relaxed),
              type, key, state, stored.tensor, recved, stored.len,
              COPY_FIRST, req_data, req_meta};
          engine_queues_[tid]->Push(msg);
        } else {  // async mode, directly add to the buffer
          CHECK_GE(bps_reducer_->sum((void*)stored.tensor, (void*)recved, len,
                                     bps_reducer_->GetDataType(stored.dtype)),
                   0);
        }
      } else {  // from other workers
        CHECK(sync_mode_);
        // CHECK(updates.merged.tensor);
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: OTHER_WORKER_SUM \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored.tensor)
                    << "\t"
                    << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                    << "len: " << len << "\t"
                    << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
        }
        if (is_engine_blocking_) {
          // TODO: decompress
          CHECK_GE(bps_reducer_->sum(
                       (void*)updates.merged.tensor, (void*)recved, len,
                       bps_reducer_->GetDataType(updates.merged.dtype)),
                   0);
        } else {  // non-blocking
          BytePSEngineMessage msg = {
              timestamp_.fetch_add(1, std::memory_order_relaxed),
              type, key, state, stored.tensor, recved, stored.len,
              SUM_RECV, req_data, req_meta};
          engine_queues_[tid]->Push(msg);
        }
      }
      // add a worker information (request.size() is the # workers received)
      updates.request.push_back(req_meta);
      SendPushResponse(state.get(), key, req_meta, server);
      if (sync_mode_ &&
          updates.request.size() == (size_t)GetExpectedWorkers(*state)) {
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: COPY_MERGED_TO_STORE \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored.tensor)
                    << "\t"
                    << "merged: "
                    << DEBUG_PRINT_TENSOR_VALUE(updates.merged.tensor) << "\t"
                    << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved);
        }
        if (is_engine_blocking_) {
          // TODO: compress
          bps_reducer_->copy(stored.tensor, updates.merged.tensor, len);
        } else {
          BytePSEngineMessage msg = {
              timestamp_.fetch_add(1, std::memory_order_relaxed),
              type, key, state, stored.tensor, stored.tensor, stored.len,
              ALL_RECV};
          engine_queues_[tid]->Push(msg);
          engine_queues_[tid]->ClearCounter(key);
        }
        updates.request.clear();
      } else if (!sync_mode_) {
        // async: clean the request buffer
        updates.request.clear();
      }
    }
  } else {  // pull request
    auto& stored = state->store;
    CHECK(stored.tensor) << "Should init the buffer for key=" << key
                          << " first";
    if (is_engine_blocking_ || !sync_mode_) {
      SendPullResponse(state.get(), type, key, req_meta, server);
    } else {
      auto tid = GetThreadID(key, 0);
      (void)tid;
      auto it = state->seen_sender.find(req_meta.sender);
      if (state->is_push_finished && (it == state->seen_sender.end())) {
        // push already finished && not received the associated pull response
        // yet
        SendPullResponse(state.get(), type, key, req_meta, server);
        state->pull_cnt += 1;
        state->seen_sender.insert(req_meta.sender);

        if (state->pull_cnt == (size_t)GetExpectedWorkers(*state)) {
          state->is_push_finished = false;
          state->pull_cnt = 0;
          state->seen_sender.clear();
        }
      } else {
        // push not finished, put into the queue, and wait for the engine
        state->q_pull_reqmeta.push_back(req_meta);
      }
    }
  }
}

void init_global_env() {
  // enable to print key profile
  log_key_info_ = GetEnv("PS_KEY_LOG", 0);

  std::string role_str = GetEnv("DMLC_ROLE", "server");
  role_ = ps::GetRole(role_str);
  if (role_str == std::string("server")) {
    is_server_ = true;
    preferred_rank = -1;
  } else {
    is_server_ = false;
    preferred_rank = 0;
  }

  LOG(INFO) << "This is a " << role_str << " is_server=" << is_server_;

  // enable engine block mode (default disabled)
  is_engine_blocking_ = GetEnv("BYTEPS_SERVER_ENGINE_BLOCKING", 0);
  if (is_engine_blocking_)
    LOG(INFO) << "Enable blocking mode of the server engine";
  record_event_ = GetEnv("DMLC_RECORD_EVENT", 0);
  if (record_event_) {
    LOG(INFO) << "Enable record trace event";
  }
  // sync or async training
  sync_mode_ = !GetEnv("BYTEPS_ENABLE_ASYNC", 0);
  if (!sync_mode_)
    LOG(INFO) << "BytePS server is enabled asynchronous training";

  // debug mode
  debug_mode_ = GetEnv("BYTEPS_SERVER_DEBUG", 0);
  debug_key_ = GetEnv("BYTEPS_SERVER_DEBUG_KEY", 0);
  if (debug_mode_)
    LOG(INFO) << "Debug mode enabled! Printing key " << debug_key_;

  // number of engine thread
  // invalid if is_engine_blocking = true
  engine_thread_num_ = GetEnv("BYTEPS_SERVER_ENGINE_THREAD", 4);
  LOG(INFO) << "BytePS server engine uses " << engine_thread_num_ << " threads"
            << ", consider increasing BYTEPS_SERVER_ENGINE_THREAD for higher "
               "performance";
  CHECK_GE(engine_thread_num_, 1);

  // enable scheduling for server engine
  enable_schedule_ = GetEnv("BYTEPS_SERVER_ENABLE_SCHEDULE", 0);
  if (enable_schedule_)
    LOG(INFO) << "Enable engine scheduling for BytePS server";

  use_global_handler_lock_ = GetEnv("BYTEPS_SERVER_GLOBAL_HANDLER_LOCK", 0);
  if (use_global_handler_lock_) {
    LOG(INFO) << "Use global BytePS handler lock compatibility mode";
  }
}

extern "C" void byteps_server() {
  init_global_env();

  // cpu reducer
  bps_reducer_ = new byteps::common::CpuReducer(nullptr);

  // init the engine
  for (size_t i = 0; i < engine_thread_num_; ++i) {
    acc_load_.push_back(0);
  }
  if (sync_mode_) {
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto q = new PriorityQueue(enable_schedule_);
      engine_queues_.push_back(q);
    }
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto t = new std::thread(&BytePSServerEngineThread, i);
      engine_threads_.push_back(t);
    }
  }

  // init server instance
  ps::StartPS(0, role_, preferred_rank, true, "byteps\0");
  byteps_server_ = new KVServer<SERVER_DATA_TYPE>(0, false, 0);
  byteps_server_->set_request_handle(BytePSHandler);
  if (!Postoffice::Get()->is_recovery()) {
    Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }

  // clean the server resource
  Finalize(0, role_, true);
  if (byteps_server_) {
    delete byteps_server_;
    byteps_server_ = nullptr;
  }
  if (bps_reducer_) {
    delete bps_reducer_;
    bps_reducer_ = nullptr;
  }
  BytePSEngineMessage msg;
  msg.ops = TERMINATE;
  for (auto q : engine_queues_) q->Push(msg);
  for (auto t : engine_threads_) t->join();

  for (auto& it : key_states_) {
    auto& state = it.second;
    if (state && state->store.tensor) {
      free(state->store.tensor);
      state->store.tensor = nullptr;
    }
  }
  key_states_.clear();

  LOG(INFO) << "byteps has been shutdown";
  return;
}

}  // namespace server
}  // namespace byteps
