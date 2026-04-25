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

#ifndef PS_RDMA_VAN_H_
#define PS_RDMA_VAN_H_

#ifdef DMLC_USE_RDMA

#include <deque>
#include <dlfcn.h>
#include <condition_variable>
#include <limits>
#include <vector>

#if (defined(DMLC_USE_CUDA) && DMLC_USE_CUDA) || \
    (defined(HAVE_CUDA) && HAVE_CUDA)
#define BYTEPS_RDMA_HAS_CUDA 1
#if defined(__has_include)
#if __has_include(<cuda.h>)
#define BYTEPS_RDMA_HAS_CUDA_DRIVER 1
#include <cuda.h>
#else
#define BYTEPS_RDMA_HAS_CUDA_DRIVER 0
#endif
#else
#define BYTEPS_RDMA_HAS_CUDA_DRIVER 1
#include <cuda.h>
#endif
#include <cuda_runtime.h>
#elif defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define BYTEPS_RDMA_HAS_CUDA 1
#if __has_include(<cuda.h>)
#define BYTEPS_RDMA_HAS_CUDA_DRIVER 1
#include <cuda.h>
#else
#define BYTEPS_RDMA_HAS_CUDA_DRIVER 0
#endif
#include <cuda_runtime.h>
#else
#define BYTEPS_RDMA_HAS_CUDA 0
#define BYTEPS_RDMA_HAS_CUDA_DRIVER 0
#endif
#else
#define BYTEPS_RDMA_HAS_CUDA 0
#define BYTEPS_RDMA_HAS_CUDA_DRIVER 0
#endif

#include "rdma_transport.h"
#include "rdma_utils.h"

namespace ps {

class RDMAVan : public Van {
 public:
  RDMAVan(Postoffice* postoffice) : Van(postoffice), postoffice_(postoffice) {
    CHECK_EQ(ibv_fork_init(), 0) << strerror(errno);
  }
  ~RDMAVan() {}

  virtual std::string GetType() const { return std::string("rdma"); }

  Postoffice* postoffice_;
  enum class RegisterRangeSource {
    kOriginal = 0,
    kCudaExact = 1,
    kProbeExpand = 2,
  };

  inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
  }

 protected:
  void RequestLocalStop() override {
    should_stop_ = true;
    recv_buffers_.Push(std::make_tuple(nullptr, nullptr));
  }

  void Start(int customer_id, bool standalone) override {
    start_mu_.lock();
    should_stop_ = false;

    auto val = Environment::Get()->find("BYTEPS_ENABLE_IPC");
    disable_ipc_ = val ? !atoi(val) : true;
    if (disable_ipc_) {
      LOG(INFO) << "Shared memory IPC has been disabled";
    } else {
      std::string role = Environment::Get()->find("DMLC_ROLE");
      if (role == "joint") {
        LOG(INFO) << "You are using IPC in joint mode, make sure no P2P "
                     "operations are involved";
      }
    }
    if (event_channel_ == nullptr) {
      event_channel_ = rdma_create_event_channel();
      CHECK(event_channel_) << "Create RDMA event channel failed";

      cm_event_polling_thread_.reset(
          new std::thread(&RDMAVan::PollEvents, this));
    }

    // enable logging
    val = Environment::Get()->find("BYTEPS_PRINT_RDMA_LOG");
    enable_log_ = val ? atoi(val) : false;
    if (enable_log_) LOG(INFO) << "Enable RDMA logging.";
    val = Environment::Get()->find("BYTEPS_RDMA_MR_DEBUG");
    enable_mr_debug_ = val ? atoi(val) : false;
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] BYTEPS_RDMA_HAS_CUDA=%d "
              "BYTEPS_RDMA_HAS_CUDA_DRIVER=%d\n",
              BYTEPS_RDMA_HAS_CUDA, BYTEPS_RDMA_HAS_CUDA_DRIVER);
      fflush(stderr);
    }
    val = Environment::Get()->find("BYTEPS_RDMA_REG_BACKOFF");
    // Default to enabled: partitioned GPU tensors often hand RDMA an interior
    // address instead of the allocation base, and retrying with a slightly
    // earlier MR base is the safe fallback when ibv_reg_mr returns EFAULT.
    enable_reg_backoff_ = val ? atoi(val) : true;

    val = Environment::Get()->find("BYTEPS_RDMA_MR_CACHE_LIMIT_MB");
    long mr_cache_limit_mb = val ? atol(val) : 192;
    mr_cache_budget_bytes_ =
        mr_cache_limit_mb > 0
            ? static_cast<size_t>(mr_cache_limit_mb) * 1024UL * 1024UL
            : 0;
    val = Environment::Get()->find("BYTEPS_RDMA_HOST_MR_CACHE_LIMIT_MB");
    long host_mr_cache_limit_mb = val ? atol(val) : 1024;
    mr_host_cache_budget_bytes_ =
        host_mr_cache_limit_mb > 0
            ? static_cast<size_t>(host_mr_cache_limit_mb) * 1024UL * 1024UL
            : 0;
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] MR cache active-byte limits device=%zu host=%zu "
              "(env BYTEPS_RDMA_MR_CACHE_LIMIT_MB / "
              "BYTEPS_RDMA_HOST_MR_CACHE_LIMIT_MB, 0 disables eviction)\n",
              mr_cache_budget_bytes_, mr_host_cache_budget_bytes_);
      fflush(stderr);
    }

    val = Environment::Get()->find("BYTEPS_RDMA_MAX_CONCURR_WR");
    if (val) {
      // should make sure: kMaxConcurrentWorkRequest >= kStartDepth +
      // kReplyDepth + kRxDepth
      kMaxConcurrentWorkRequest = atoi(val);

      auto start_depth_env =
          Environment::Get()->find("BYTEPS_RDMA_START_DEPTH");
      auto rx_depth_env = Environment::Get()->find("BYTEPS_RDMA_RX_DEPTH");
      auto ctrl_rx_depth_env =
          Environment::Get()->find("BYTEPS_RDMA_CTRL_RX_DEPTH");

      auto start_depth = start_depth_env ? atoi(start_depth_env) : 128;
      auto rx_depth = rx_depth_env ? atoi(rx_depth_env) : 2048;
      auto control_rx_depth =
          ctrl_rx_depth_env ? atoi(ctrl_rx_depth_env) : std::min(rx_depth, 128);
      auto reply_depth = std::max(rx_depth, control_rx_depth);

      CHECK_GE(kMaxConcurrentWorkRequest, start_depth + reply_depth + rx_depth)
          << "Should make sure: kMaxConcurrentWorkRequest >= kStartDepth + "
             "kReplyDepth + kRxDepth";
    }

    start_mu_.unlock();
    if (!standalone) Van::Start(customer_id, false);
  }

  void Stop() override {
    PS_VLOG(1) << my_node_.ShortDebugString() << " is stopping";
    Van::Stop();

    CHECK(should_stop_);

    PS_VLOG(1) << "Stopping cq_polling_thread_.";
    if (cq_polling_thread_) {
      cq_polling_thread_->join();
      cq_polling_thread_.reset();
    }

    PS_VLOG(1) << "Stopping cm_event_polling_thread_.";
    if (cm_event_polling_thread_) {
      cm_event_polling_thread_->join();
      cm_event_polling_thread_.reset();
    }

    PS_VLOG(1) << "Clearing memory allocator.";
    mem_allocator_.reset();

    PS_VLOG(1) << "Clearing endpoints.";
    {
      std::lock_guard<std::mutex> lk(incoming_mu_);
      incoming_.clear();
    }
    {
      std::lock_guard<std::mutex> lk(endpoints_mu_);
      endpoints_.clear();
      data_endpoints_.clear();
    }
    {
      std::lock_guard<std::mutex> lk(qp_map_mu_);
      qp_num_to_endpoint_.clear();
    }
    if (shared_rx_ctx_) {
      for (int i = 0; i < srq_depth_; ++i) {
        if (!(shared_rx_ctx_[i].buffer)) {
          continue;
        }
        free(shared_rx_ctx_[i].buffer->addr);
        CHECK_EQ(ibv_dereg_mr(shared_rx_ctx_[i].buffer), 0);
      }
      delete[] shared_rx_ctx_;
      shared_rx_ctx_ = nullptr;
    }
    if (srq_) {
      CHECK(!ibv_destroy_srq(srq_)) << "Failed to destroy SRQ";
      srq_ = nullptr;
    }
    PS_VLOG(1) << "Destroying cq and pd.";
    CHECK(!ibv_destroy_cq(cq_)) << "Failed to destroy CQ";
    CHECK(!ibv_destroy_comp_channel(comp_event_channel_))
        << "Failed to destroy channel";

    for (auto& it : mem_mr_) ibv_dereg_mr(it.second);

    // TODO: ibv_dealloc_pd sometimes complains resource busy, need to fix this
    // CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD: " <<
    // strerror(errno);

    PS_VLOG(1) << "Destroying listener.";
    rdma_destroy_id(listener_);
    rdma_destroy_event_channel(event_channel_);
  }

  int Bind(Node& node, int max_retry) override {
    CHECK_EQ(my_node_.num_ports, 1)
        << "RDMA van does not support multiple ports";
    CHECK(rdma_create_id(event_channel_, &listener_, nullptr, RDMA_PS_TCP) == 0)
        << "Create RDMA connection identifier failed";

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));

    auto val = Environment::Get()->find("DMLC_NODE_HOST");
    if (val) {
      PS_VLOG(1) << "bind to DMLC_NODE_HOST: " << std::string(val);
      addr.sin_addr.s_addr = inet_addr(val);
    }

    addr.sin_family = AF_INET;
    int port = node.port;
    unsigned seed = static_cast<unsigned>(time(NULL) + port);
    for (int i = 0; i < max_retry + 1; ++i) {
      addr.sin_port = htons(port);
      if (rdma_bind_addr(listener_,
                         reinterpret_cast<struct sockaddr*>(&addr)) == 0) {
        break;
      }
      if (i == max_retry) {
        port = -1;
      } else {
        port = 10000 + rand_r(&seed) % 40000;
      }
    }
    CHECK(rdma_listen(listener_, kRdmaListenBacklog) == 0)
        << "Listen RDMA connection failed: " << strerror(errno);
    return port;
  }

  void Connect2Node(const Node& node, bool dataPlane = false) {
    PS_VLOG(1) << "Connecting to Node " << node.id
               << ", My_Node=" << my_node_.id;
    CHECK_NE(node.id, node.kEmpty);
    CHECK_NE(node.port, node.kEmpty);
    CHECK(node.hostname.size());

    // worker doesn't need to connect to the other workers. same for server
    if ((node.role == my_node_.role) && (node.id != my_node_.id)) {
      return;
    }

    if (node.id != Node::kEmpty) {
      endpoints_mu_.lock();
      auto& whichEndpoints = dataPlane ? data_endpoints_ : endpoints_;

      auto it = whichEndpoints.find(node.id);

      // if there is an endpoint with pending connection
      if (it != whichEndpoints.end()) {
        UnregisterEndpointQP(it->second.get());
        whichEndpoints.erase(it);
      }

      Endpoint* endpoint;
      whichEndpoints[node.id] = std::make_unique<Endpoint>(dataPlane);
      endpoint = whichEndpoints[node.id].get();

      endpoints_mu_.unlock();
      endpoint->SetNodeID(node.id);

      struct addrinfo* remote_addr;
      CHECK_EQ(
          getaddrinfo(node.hostname.c_str(), std::to_string(node.port).c_str(),
                      nullptr, &remote_addr),
          0);

      while (endpoint->status != Endpoint::CONNECTED) {
        std::unique_lock<std::mutex> lk(endpoint->connect_mu);
        endpoint->status = Endpoint::CONNECTING;

        if (endpoint->cm_id != nullptr) {
          rdma_destroy_qp(endpoint->cm_id);
          CHECK_EQ(rdma_destroy_id(endpoint->cm_id), 0) << strerror(errno);
          endpoint->cm_id = nullptr;
        }

        CHECK_EQ(rdma_create_id(event_channel_, &endpoint->cm_id, nullptr,
                                RDMA_PS_TCP),
                 0)
            << "Create RDMA connection identifier failed";
        endpoint->cm_id->context = endpoint;

        auto val = Environment::Get()->find("DMLC_NODE_HOST");
        if (val) {
          struct addrinfo* addr;
          auto rc = getaddrinfo(val, "", NULL, &addr);
          CHECK_EQ(rc, 0) << "getaddrinfo failed: " << gai_strerror(rc);

          CHECK_EQ(rdma_resolve_addr(endpoint->cm_id, addr->ai_addr,
                                     remote_addr->ai_addr, kTimeoutms),
                   0)
              << "Resolve RDMA address failed with errno: " << strerror(errno);
        } else {
          CHECK_EQ(rdma_resolve_addr(endpoint->cm_id, nullptr,
                                     remote_addr->ai_addr, kTimeoutms),
                   0)
              << "Resolve RDMA address failed with errno: " << strerror(errno);
        }

        endpoint->cv.wait(lk, [endpoint] {
          return endpoint->status != Endpoint::CONNECTING;
        });

        if (endpoint->status == Endpoint::CONNECTED) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }

      bool is_local_node =
          disable_ipc_ ? false
                       : (node.hostname == my_node_.hostname ? true : false);
      {
        std::lock_guard<std::mutex> lk(local_mu_);
        is_local_[node.id] = is_local_node;
      }

      LOG(INFO) << "Connect to Node " << node.id
                << " with Transport=" << (is_local_node ? "IPC" : "RDMA");

      std::shared_ptr<Transport> t =
          is_local_node ? std::make_shared<IPCTransport>(
                              endpoint, mem_allocator_.get(), postoffice_)
                        : std::make_shared<RDMATransport>(
                              endpoint, mem_allocator_.get(), postoffice_);
      endpoint->SetTransport(t);

      freeaddrinfo(remote_addr);
    }
  }

  void Connect(const Node& node) override {
    if (node.id == my_node_.id) {
      return;
    }
    Connect2Node(node, false);

    // Only create data plane connections between workers and servers
    // Scheduler should only use control plane connections
    // Servers/workers should not create data plane connections to scheduler
    if (my_node_.role == Node::WORKER && node.role == Node::SERVER) {
      Connect2Node(node, true);
    }
  }

  Endpoint* FindIncomingEndpoint(int remote_id, bool data_plane) {
    std::lock_guard<std::mutex> lk(incoming_mu_);
    for (const auto& endpoint_holder : incoming_) {
      Endpoint* endpoint = endpoint_holder.get();
      if (endpoint->node_id != remote_id) {
        continue;
      }
      if (endpoint->isDataPlane != data_plane) {
        continue;
      }
      if (endpoint->status != Endpoint::CONNECTED) {
        continue;
      }
      if (!endpoint->GetTransport()) {
        continue;
      }
      return endpoint;
    }
    return nullptr;
  }
  // void Connect(const Node &node) override {
  //   PS_VLOG(1) << "Connecting to Node " << node.id
  //              << ", My_Node=" << my_node_.id;
  //   CHECK_NE(node.id, node.kEmpty);
  //   CHECK_NE(node.port, node.kEmpty);
  //   CHECK(node.hostname.size());

  //   // worker doesn't need to connect to the other workers. same for server
  //   if ((node.role == my_node_.role) && (node.id != my_node_.id)) {
  //     return;
  //   }

  //   if (node.id != Node::kEmpty) {
  //     endpoints_mu_.lock();
  //     auto it = endpoints_.find(node.id);

  //     // if there is an endpoint with pending connection
  //     if (it != endpoints_.end()) {
  //       endpoints_.erase(it);
  //     }

  //     Endpoint *endpoint;
  //     endpoints_[node.id] = std::make_unique<Endpoint>();
  //     endpoint = endpoints_[node.id].get();
  //     endpoints_mu_.unlock();

  //     endpoint->SetNodeID(node.id);

  //     struct addrinfo *remote_addr;
  //     CHECK_EQ(
  //         getaddrinfo(node.hostname.c_str(),
  //         std::to_string(node.port).c_str(),
  //                     nullptr, &remote_addr),
  //         0);

  //     while (endpoint->status != Endpoint::CONNECTED) {
  //       std::unique_lock<std::mutex> lk(endpoint->connect_mu);
  //       endpoint->status = Endpoint::CONNECTING;

  //       if (endpoint->cm_id != nullptr) {
  //         rdma_destroy_qp(endpoint->cm_id);
  //         CHECK_EQ(rdma_destroy_id(endpoint->cm_id), 0) << strerror(errno);
  //         endpoint->cm_id = nullptr;
  //       }

  //       CHECK_EQ(rdma_create_id(event_channel_, &endpoint->cm_id, nullptr,
  //                               RDMA_PS_TCP),
  //                0)
  //           << "Create RDMA connection identifier failed";
  //       endpoint->cm_id->context = endpoint;

  //       auto val = Environment::Get()->find("DMLC_NODE_HOST");
  //       if (val) {
  //         struct addrinfo *addr;
  //         auto rc = getaddrinfo(val, "", NULL, &addr);
  //         CHECK_EQ(rc, 0) << "getaddrinfo failed: " << gai_strerror(rc);

  //         CHECK_EQ(rdma_resolve_addr(endpoint->cm_id, addr->ai_addr,
  //                                    remote_addr->ai_addr, kTimeoutms),
  //                  0)
  //             << "Resolve RDMA address failed with errno: " <<
  //             strerror(errno);
  //       } else {
  //         CHECK_EQ(rdma_resolve_addr(endpoint->cm_id, nullptr,
  //                                    remote_addr->ai_addr, kTimeoutms),
  //                  0)
  //             << "Resolve RDMA address failed with errno: " <<
  //             strerror(errno);
  //       }

  //       endpoint->cv.wait(lk, [endpoint] {
  //         return endpoint->status != Endpoint::CONNECTING;
  //       });

  //       if (endpoint->status == Endpoint::CONNECTED) break;
  //       std::this_thread::sleep_for(std::chrono::milliseconds(500));
  //     }

  //     bool is_local_node =
  //         disable_ipc_ ? false
  //                      : (node.hostname == my_node_.hostname ? true : false);
  //     {
  //       std::lock_guard<std::mutex> lk(local_mu_);
  //       is_local_[node.id] = is_local_node;
  //     }

  //     LOG(INFO) << "Connect to Node " << node.id
  //               << " with Transport=" << (is_local_node ? "IPC" : "RDMA");

  //     std::shared_ptr<Transport> t =
  //         is_local_node ? std::make_shared<IPCTransport>(
  //                             endpoint, mem_allocator_.get(), postoffice_)
  //                       : std::make_shared<RDMATransport>(
  //                             endpoint, mem_allocator_.get(), postoffice_);
  //     endpoint->SetTransport(t);

  //     freeaddrinfo(remote_addr);
  //   }
  // }

  int SendMsg(Message& msg) override {
    int remote_id = msg.meta.recver;
    CHECK_NE(remote_id, Meta::kEmpty);
    bool is_pushpull = IsValidPushpull(msg);

    endpoints_mu_.lock();
    Endpoint* endpoint = nullptr;
    auto endpoint_it = endpoints_.find(remote_id);
    if (endpoint_it != endpoints_.end()) {
      endpoint = endpoint_it->second.get();
    }
    Endpoint* dataEndpoint = nullptr;
    if (is_pushpull) {
      auto data_it = data_endpoints_.find(remote_id);
      if (data_it != data_endpoints_.end()) {
        dataEndpoint = data_it->second.get();
      }
    }
    endpoints_mu_.unlock();

    if (endpoint == nullptr) {
      endpoint = FindIncomingEndpoint(remote_id, false);
    }
    CHECK(endpoint != nullptr)
        << "Control endpoint not ready for remote_id=" << remote_id
        << ", local_id=" << my_node_.id;
    if (is_pushpull && dataEndpoint == nullptr) {
      dataEndpoint = FindIncomingEndpoint(remote_id, true);
    }
    CHECK(!is_pushpull || dataEndpoint != nullptr)
        << "Data endpoint not ready for remote_id=" << remote_id
        << ", local_id=" << my_node_.id;

    int meta_len = GetPackMetaLen(msg.meta);
    size_t data_len = msg.meta.data_size;
    size_t total_len = meta_len + data_len;
    CHECK(meta_len);

    // Decode key before MR registration so throttling/debug paths can report
    // the real tensor key instead of the default zero value.
    if (is_pushpull) {
      AddMeta(msg);
    } else {
      RegisterMemory(msg, true);
    }

    auto trans = CHECK_NOTNULL(endpoint->GetTransport());
    std::shared_ptr<Transport> dataTrans;
    if (is_pushpull) {
      dataTrans = CHECK_NOTNULL(dataEndpoint->GetTransport());
    }

    // start rendezvous if no remote info
    if (!is_pushpull) {
      MessageBuffer* msg_buf = PrepareNewMsgBuf(msg);
      StoreMsgBuf(msg_buf, msg);
      trans->SendRendezvousBegin(msg, msg_buf);
      return total_len;

    } else {
      auto is_push = msg.meta.push;
      auto key = msg.meta.key;
      if (!HasRemoteInfo(msg, key, is_push, remote_id)) {
        MessageBuffer* msg_buf = PrepareNewMsgBuf(msg);
        PrepareRendezvousMR(msg, msg_buf);
        StoreMsgBuf(msg_buf, msg);
        trans->SendRendezvousBegin(msg, msg_buf);
        return total_len;
      }
    }

    auto addr_tuple =
        GetRemoteAndLocalInfo(msg.meta.key, msg.meta.push, remote_id);
    MessageBuffer* msg_buf = std::get<3>(addr_tuple);  // local message buffer

    // prepare new meta and data
    CHECK_EQ(msg_buf->inline_len, (size_t)meta_len);
    CHECK(msg_buf->inline_buf);
    msg_buf->data = msg.data;
    if (msg.meta.push && msg.meta.request) {
      // IMPORTANT: msg_buf is reused across steps for the same key.
      // The payload pointer may change, so MR cache in msg_buf must be
      // refreshed to keep SGE address and lkey from the same memory region.
      msg_buf->mrs.clear();
      PrepareData(msg, msg_buf, true);
      if (enable_mr_debug_) {
        CHECK_EQ(msg_buf->mrs.size(), 1);
        fprintf(stderr,
                "[MR-DEBUG] refresh key=%llu data=%p len=%zu mr.addr=%p "
                "mr.len=%u lkey=%u\n",
                (unsigned long long)msg.meta.key, msg_buf->data[1].data(),
                msg_buf->data[1].size(), msg_buf->mrs[0].first->addr,
                msg_buf->mrs[0].first->length, msg_buf->mrs[0].first->lkey);
        fflush(stderr);
      }
    } else if (!msg.meta.push && msg.meta.request) {
      msg_buf->mrs.clear();
      msg_buf->release_mrs_on_completion = false;
      auto keys =
          HoldOrRegisterMR(msg, msg_buf, reinterpret_cast<char*>(msg.meta.addr),
                           msg.meta.val_len, "pull-request", true);
      msg.meta.option = keys.rkey;
    }
    RepackMsgBufMeta(msg, msg_buf);

    PrintSendLog(msg, msg_buf, addr_tuple);

    // already know remote address, directly use RDMA-write
    if (msg.meta.push && msg.meta.request) {
      // worker, push request
      msg_buf->release_mrs_on_completion = true;
      dataTrans->SendPushRequest(msg, msg_buf, addr_tuple);
    } else if (msg.meta.push && !msg.meta.request) {
      // server, push response
      msg_buf->release_mrs_on_completion = true;
      trans->SendPushResponse(msg, msg_buf, addr_tuple);
    } else if (!msg.meta.push && msg.meta.request) {
      // worker, pull request
      trans->SendPullRequest(msg, msg_buf, addr_tuple);
    } else if (!msg.meta.push && !msg.meta.request) {
      // server, pull response
      msg_buf->mrs.clear();
      msg_buf->release_mrs_on_completion = true;
      auto keys = HoldOrRegisterMR(msg, msg_buf, msg_buf->data[1].data(),
                                   msg_buf->data[1].size(), "pull-response",
                                   true);
      dataTrans->SendPullResponse(msg, msg_buf, addr_tuple, keys.lkey);
    } else {
      CHECK(0) << "unexpected message type";
    }

    return total_len;
  }

  int RecvMsg(Message* msg) override {
    msg->data.clear();
    std::tuple<Endpoint*, BufferContext*> notification;
    recv_buffers_.WaitAndPop(&notification);

    Endpoint* endpoint = std::get<Endpoint*>(notification);
    BufferContext* buffer_ctx = std::get<BufferContext*>(notification);
    if (endpoint == nullptr || buffer_ctx == nullptr) {
      msg->meta = Meta();
      msg->meta.recver = my_node_.id;
      msg->meta.sender = my_node_.id;
      msg->meta.control.cmd = Control::TERMINATE;
      return 0;
    }

    msg->meta.recver = my_node_.id;
    msg->meta.sender = endpoint->node_id;

    // the second argument is actually deprecated,
    // we keep it as is in order to be compatible
    UnpackMeta(buffer_ctx->buffer, buffer_ctx->meta_len, &msg->meta);
    int meta_len = GetPackMetaLen(msg->meta);

    int total_len = 0;
    total_len += meta_len;

    auto trans = CHECK_NOTNULL(endpoint->GetTransport());

    PrintRecvLog(msg, buffer_ctx, meta_len);

    if (!IsValidPushpull(*msg)) {
      return total_len;
    }

    // valid data message
    if (msg->meta.push && msg->meta.request) {
      // push request
      total_len += trans->RecvPushRequest(msg, buffer_ctx, meta_len);
    } else if (!msg->meta.push && msg->meta.request) {
      // pull request
      total_len += trans->RecvPullRequest(msg, buffer_ctx, meta_len);
    } else if (msg->meta.push && !msg->meta.request) {
      // push response
      total_len += trans->RecvPushResponse(msg, buffer_ctx, meta_len);
    } else if (!msg->meta.push && !msg->meta.request) {
      // pull response
      total_len += trans->RecvPullResponse(msg, buffer_ctx, meta_len);
      ReleasePullRequestMR(msg->meta.key, msg->meta.sender);
    } else {
      CHECK(0) << "unknown msg type";
    }

    return total_len;
  }

 private:
  void PrintSendLog(Message& msg, MessageBuffer* msg_buf,
                    RemoteTuple remote_tuple) {
    if (!enable_log_) return;
    std::lock_guard<std::mutex> lock(log_mu_);

    if (!IsValidPushpull(msg)) {
      LOG(INFO) << "Send Control Message" << std::flush;
    } else if (msg.meta.push && msg.meta.request) {
      // worker, push request
      LOG(INFO) << "Send Push Request: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t tensor_len=" << msg_buf->mrs[0].second
                << "\t remote_idx=" << std::get<2>(remote_tuple)
                << "\t remote_addr=" << (void*)std::get<0>(remote_tuple)
                << std::flush;
    } else if (msg.meta.push && !msg.meta.request) {
      // server, push response
      LOG(INFO) << "Send Push Response: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t remote_idx=" << std::get<2>(remote_tuple)
                << "\t remote_addr=" << (void*)std::get<0>(remote_tuple)
                << std::flush;
    } else if (!msg.meta.push && msg.meta.request) {
      // worker, pull request
      LOG(INFO) << "Send Pull Request: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t remote_idx=" << std::get<2>(remote_tuple)
                << "\t remote_addr=" << (void*)std::get<0>(remote_tuple)
                << std::flush;
    } else if (!msg.meta.push && !msg.meta.request) {
      // server, pull response
      LOG(INFO) << "Send Pull Response: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t tensor_len=" << msg.meta.val_len << "\t idx="
                << "none"
                << "\t remote_addr=" << (void*)msg.meta.addr << std::flush;
    }
  }

  void PrintRecvLog(Message* msg, BufferContext* buffer_ctx, int meta_len) {
    if (!enable_log_) return;
    std::lock_guard<std::mutex> lock(log_mu_);

    if (!IsValidPushpull(*msg)) {
      LOG(INFO) << "Recv Control Message" << std::flush;
    } else if (msg->meta.push && msg->meta.request) {
      // push request
      LOG(INFO) << "Recv Push Request: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender
                << "\t tensor_len=" << buffer_ctx->data_len[1] << std::flush;
    } else if (!msg->meta.push && msg->meta.request) {
      // pull request
      LOG(INFO) << "Recv Pull Request: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender << std::flush;
    } else if (msg->meta.push && !msg->meta.request) {
      // push response
      LOG(INFO) << "Recv Push Response: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender << std::flush;
    } else if (!msg->meta.push && !msg->meta.request) {
      // pull response
      LOG(INFO) << "Recv Pull Response: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender
                << "\t tensor_len=" << msg->meta.val_len;
    }
  }

  bool HasRemoteInfo(Message& msg, uint64_t key, bool is_push, int recver) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    if (is_push && (push_addr_.find(key) != push_addr_.end()) &&
        (push_addr_[key].find(recver) != push_addr_[key].end())) {
      // Check if the cached MR is large enough for the current push data
      // This handles the case where a small push (e.g., kGroupRegister with 4
      // bytes) is followed by a large push (e.g., kDefaultPushPull with MB of
      // data) for the same key, which would otherwise cause a size mismatch
      // crash.
      auto& addr_tuple = push_addr_[key][recver];
      MessageBuffer* cached_msg_buf = std::get<3>(addr_tuple);
      if ((cached_msg_buf->mrs.size() > 0 || cached_msg_buf->data.size() > 1) &&
          msg.data.size() > 1) {
        size_t cached_mr_size =
            cached_msg_buf->mrs.size() > 0 ? cached_msg_buf->mrs[0].second
                                           : cached_msg_buf->data[1].size();
        size_t current_data_size = msg.data[1].size();
        if (current_data_size > cached_mr_size) {
          // Cached MR is too small, invalidate and force re-rendezvous
          push_addr_[key].erase(recver);
          return false;
        }
      }
      return true;
    }
    if (!is_push && (pull_addr_.find(key) != pull_addr_.end()) &&
        (pull_addr_[key].find(recver) != pull_addr_[key].end())) {
      return true;
    }

    return false;
  }

  void StoreMsgBuf(MessageBuffer* msg_buf, Message& msg) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    CHECK_EQ(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());
    msgbuf_cache_[msg_buf] = msg;
  }

  Message* GetFirstMsg(MessageBuffer* msg_buf) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    CHECK_NE(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());
    return &msgbuf_cache_[msg_buf];
  }

  void ReleaseFirstMsg(MessageBuffer* msg_buf) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    CHECK_NE(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());
    msgbuf_cache_.erase(msg_buf);
  }

  void StoreRemoteAndLocalInfo(MessageBuffer* msg_buf, uint64_t remote_addr,
                               uint32_t rkey, uint32_t idx) {
    std::lock_guard<std::mutex> lk(addr_mu_);

    CHECK_NE(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());

    auto& msg = msgbuf_cache_[msg_buf];

    auto key = msg.meta.key;
    auto is_push = msg.meta.push;
    auto recver = msg.meta.recver;

    auto t = std::make_tuple(remote_addr, rkey, idx, msg_buf);
    if (is_push) {
      push_addr_[key][recver] = t;
    } else {
      pull_addr_[key][recver] = t;
    }
  }

  RemoteTuple GetRemoteAndLocalInfo(uint64_t key, bool is_push, int recver) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    return (is_push ? push_addr_[key][recver] : pull_addr_[key][recver]);
  }

  MessageBuffer* PrepareNewMsgBuf(Message& msg) {
    MessageBuffer* msg_buf = new MessageBuffer();
    auto meta_len = GetPackMetaLen(msg.meta);
    msg_buf->inline_len = meta_len;
    msg_buf->inline_buf = mem_allocator_->Alloc(meta_len);
    msg_buf->data = msg.data;
    PackMeta(msg.meta, &(msg_buf->inline_buf), &meta_len);
    return msg_buf;
  }

  enum class MrCacheKind { kHost, kDevice };

  struct CachedMrInfo {
    size_t bytes = 0;
    size_t refs = 0;
    uint64_t last_used = 0;
    MrCacheKind kind = MrCacheKind::kHost;
  };

  struct HeldMrKeys {
    uint32_t lkey = 0;
    uint32_t rkey = 0;
  };

  const char* MrCacheKindName(MrCacheKind kind) const {
    switch (kind) {
      case MrCacheKind::kDevice:
        return "device";
      case MrCacheKind::kHost:
        return "host";
    }
    return "unknown";
  }

  size_t MrCacheBudgetBytes(MrCacheKind kind) const {
    return kind == MrCacheKind::kDevice ? mr_cache_budget_bytes_
                                        : mr_host_cache_budget_bytes_;
  }

  size_t& MrCacheActiveBytes(MrCacheKind kind) {
    return kind == MrCacheKind::kDevice ? mr_device_cache_active_bytes_
                                        : mr_host_cache_active_bytes_;
  }

  void AddMrCacheBytes(MrCacheKind kind, size_t bytes) {
    mr_cache_active_bytes_ += bytes;
    MrCacheActiveBytes(kind) += bytes;
  }

  void SubMrCacheBytes(MrCacheKind kind, size_t bytes) {
    if (mr_cache_active_bytes_ >= bytes) {
      mr_cache_active_bytes_ -= bytes;
    } else {
      mr_cache_active_bytes_ = 0;
    }
    auto& active = MrCacheActiveBytes(kind);
    if (active >= bytes) {
      active -= bytes;
    } else {
      active = 0;
    }
  }

  void RemoveCachedMRFromLruLocked(struct ibv_mr* mr) {
    for (auto it = mr_lru_.begin(); it != mr_lru_.end();) {
      if (*it == mr) {
        it = mr_lru_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void TouchCachedMRLocked(struct ibv_mr* mr) {
    auto it = mr_cache_info_.find(mr);
    if (it == mr_cache_info_.end()) return;
    it->second.last_used = ++mr_lru_clock_;
    RemoveCachedMRFromLruLocked(mr);
    mr_lru_.push_back(mr);
  }

  void DeregisterCachedMRLocked(struct ibv_mr* mr, const char* reason) {
    auto info_it = mr_cache_info_.find(mr);
    if (info_it == mr_cache_info_.end()) return;
    CHECK_EQ(info_it->second.refs, 0U)
        << "Attempted to evict an in-use MR";
    size_t bytes = info_it->second.bytes;
    MrCacheKind kind = info_it->second.kind;
    RemoveCachedMRFromLruLocked(mr);
    auto base = reinterpret_cast<char*>(mr->addr);
    auto map_it = mem_mr_.find(base);
    if (map_it != mem_mr_.end() && map_it->second == mr) {
      mem_mr_.erase(map_it);
    }
    mr_cache_info_.erase(info_it);
    SubMrCacheBytes(kind, bytes);
    mr_cache_cv_.notify_all();
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] evict MR reason=%s kind=%s addr=%p len=%zu "
              "kind_active=%zu kind_limit=%zu total_active=%zu\n",
              reason, MrCacheKindName(kind), mr->addr, bytes,
              MrCacheActiveBytes(kind), MrCacheBudgetBytes(kind),
              mr_cache_active_bytes_);
      fflush(stderr);
    }
    CHECK_EQ(ibv_dereg_mr(mr), 0) << "ibv_dereg_mr failed: " << strerror(errno);
  }

  bool EvictOneCachedMRLocked(const char* reason, MrCacheKind kind) {
    struct ibv_mr* victim = nullptr;
    for (auto* mr : mr_lru_) {
      auto info_it = mr_cache_info_.find(mr);
      if (info_it == mr_cache_info_.end()) continue;
      if (info_it->second.kind != kind) continue;
      if (info_it->second.refs != 0) continue;
      victim = mr;
      break;
    }
    if (victim == nullptr) return false;
    DeregisterCachedMRLocked(victim, reason);
    return true;
  }

  bool MrCoversRange(struct ibv_mr* mr, char* ptr, size_t len) const {
    if (mr == nullptr || ptr == nullptr) return false;
    auto mr_base = reinterpret_cast<uintptr_t>(mr->addr);
    auto mr_end = mr_base + mr->length;
    auto p = reinterpret_cast<uintptr_t>(ptr);
    auto pend = p + len;
    return p >= mr_base && pend <= mr_end;
  }

  bool EnsureMRCacheBudgetLocked(MrCacheKind kind, size_t incoming_bytes,
                                 uint64_t key,
                                 const char* label,
                                 std::unique_lock<std::mutex>* lock,
                                 bool wait_for_budget) {
    size_t budget = MrCacheBudgetBytes(kind);
    if (budget == 0) return true;
    if (incoming_bytes > budget) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] MR cache request exceeds budget key=%llu "
                "label=%s kind=%s incoming=%zu limit=%zu\n",
                (unsigned long long)key, label, MrCacheKindName(kind),
                incoming_bytes, budget);
        fflush(stderr);
      }
      errno = ENOMEM;
      return false;
    }
    while (MrCacheActiveBytes(kind) + incoming_bytes > budget) {
      if (!EvictOneCachedMRLocked(label, kind)) {
        if (enable_mr_debug_) {
          fprintf(stderr,
                  "[MR-DEBUG] MR cache budget saturated key=%llu label=%s "
                  "kind=%s incoming=%zu kind_active=%zu limit=%zu "
                  "total_active=%zu; no unreferenced victim%s\n",
                  (unsigned long long)key, label, MrCacheKindName(kind),
                  incoming_bytes, MrCacheActiveBytes(kind), budget,
                  mr_cache_active_bytes_,
                  wait_for_budget ? ", waiting" : "");
          fflush(stderr);
        }
        if (!wait_for_budget || lock == nullptr) {
          errno = EAGAIN;
          return false;
        }
        mr_cache_cv_.wait(*lock);
      }
    }
    return true;
  }

  bool ShouldRetryMrRegistrationAfterEvict(int err) const {
    return err == EFAULT || err == ENOMEM || err == EAGAIN;
  }

  struct ibv_mr* RegisterMRRawWithResourceRetryLocked(
      char* ptr, size_t len, int flags, uint64_t key, const char* label,
      MrCacheKind kind, std::unique_lock<std::mutex>* lock,
      bool wait_for_budget) {
    while (true) {
      errno = 0;
      auto* mr = ibv_reg_mr(mem_allocator_->GetPD(), ptr, len, flags);
      if (mr != nullptr) return mr;

      int reg_errno = errno;
      if (kind != MrCacheKind::kDevice ||
          !ShouldRetryMrRegistrationAfterEvict(reg_errno)) {
        errno = reg_errno;
        return nullptr;
      }

      if (EvictOneCachedMRLocked(label, kind)) {
        if (enable_mr_debug_) {
          fprintf(stderr,
                  "[MR-DEBUG] retry device MR reg after evict key=%llu "
                  "label=%s ptr=%p len=%zu err=%s\n",
                  (unsigned long long)key, label, ptr, len,
                  strerror(reg_errno));
          fflush(stderr);
        }
        continue;
      }

      if (!wait_for_budget || lock == nullptr) {
        errno = reg_errno;
        return nullptr;
      }

      if (MrCacheActiveBytes(kind) == 0) {
        errno = reg_errno;
        return nullptr;
      }

      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] device MR reg failed with no victim key=%llu "
                "label=%s ptr=%p len=%zu err=%s, waiting\n",
                (unsigned long long)key, label, ptr, len, strerror(reg_errno));
        fflush(stderr);
      }
      mr_cache_cv_.wait(*lock);
    }
  }

  struct ibv_mr* RegisterMRWithBudgetLocked(char* ptr, size_t len, int flags,
                                            uint64_t key, const char* label,
                                            MrCacheKind kind,
                                            std::unique_lock<std::mutex>* lock,
                                            bool wait_for_budget) {
    while (true) {
      auto* cached = FindMRByRangeLocked(ptr, len);
      if (cached != nullptr) {
        return cached;
      }
      auto same_base_it = mem_mr_.find(ptr);
      if (same_base_it == mem_mr_.end()) break;
      auto* same_base_mr = same_base_it->second;
      auto same_info = mr_cache_info_.find(same_base_mr);
      CHECK(same_info != mr_cache_info_.end());
      if (MrCoversRange(same_base_mr, ptr, len) ||
          same_info->second.refs == 0) {
        break;
      }
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] same-base MR in use key=%llu label=%s "
                "addr=%p len=%zu old.len=%zu old.refs=%zu%s\n",
                (unsigned long long)key, label, ptr, len,
                static_cast<size_t>(same_base_mr->length),
                same_info->second.refs,
                wait_for_budget ? ", waiting" : "");
        fflush(stderr);
      }
      if (!wait_for_budget || lock == nullptr) {
        errno = EAGAIN;
        return nullptr;
      }
      mr_cache_cv_.wait(*lock);
    }
    if (!EnsureMRCacheBudgetLocked(kind, len, key, label, lock,
                                   wait_for_budget)) {
      return nullptr;
    }
    while (true) {
      auto* cached = FindMRByRangeLocked(ptr, len);
      if (cached != nullptr) {
        return cached;
      }
      auto same_base_it = mem_mr_.find(ptr);
      if (same_base_it == mem_mr_.end()) break;
      auto* same_base_mr = same_base_it->second;
      auto same_info = mr_cache_info_.find(same_base_mr);
      CHECK(same_info != mr_cache_info_.end());
      if (MrCoversRange(same_base_mr, ptr, len) ||
          same_info->second.refs == 0) {
        break;
      }
      if (!wait_for_budget || lock == nullptr) {
        errno = EAGAIN;
        return nullptr;
      }
      mr_cache_cv_.wait(*lock);
    }
    return RegisterMRRawWithResourceRetryLocked(ptr, len, flags, key, label,
                                                kind, lock, wait_for_budget);
  }

  void InsertCachedMRLocked(struct ibv_mr* mr, uint64_t key,
                            const char* label, MrCacheKind kind) {
    CHECK(mr);
    auto base = reinterpret_cast<char*>(mr->addr);
    auto old_it = mem_mr_.find(base);
    if (old_it != mem_mr_.end() && old_it->second != mr) {
      auto old_info = mr_cache_info_.find(old_it->second);
      CHECK(old_info != mr_cache_info_.end());
      CHECK_EQ(old_info->second.refs, 0U) << "Replacing an in-use MR";
      DeregisterCachedMRLocked(old_it->second, "replace");
    }
    mem_mr_[base] = mr;
    if (mr_cache_info_.find(mr) == mr_cache_info_.end()) {
      CachedMrInfo info;
      info.bytes = static_cast<size_t>(mr->length);
      info.kind = kind;
      mr_cache_info_[mr] = info;
      AddMrCacheBytes(kind, info.bytes);
    }
    TouchCachedMRLocked(mr);
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] cache MR key=%llu label=%s kind=%s addr=%p len=%zu "
              "kind_active=%zu kind_limit=%zu total_active=%zu\n",
              (unsigned long long)key, label, MrCacheKindName(kind), mr->addr,
              static_cast<size_t>(mr->length), MrCacheActiveBytes(kind),
              MrCacheBudgetBytes(kind), mr_cache_active_bytes_);
      fflush(stderr);
    }
  }

  MRPtr MakeCachedMRRefLocked(struct ibv_mr* mr) {
    auto it = mr_cache_info_.find(mr);
    CHECK(it != mr_cache_info_.end()) << "MR is not in cache";
    ++it->second.refs;
    TouchCachedMRLocked(mr);
    return MRPtr(mr, [this](struct ibv_mr* mr) { ReleaseCachedMR(mr); });
  }

  void ReleaseCachedMR(struct ibv_mr* mr) {
    if (mr == nullptr) return;
    std::lock_guard<std::mutex> lock(map_mu_);
    auto it = mr_cache_info_.find(mr);
    if (it == mr_cache_info_.end()) return;
    CHECK_GT(it->second.refs, 0U);
    --it->second.refs;
    TouchCachedMRLocked(mr);
    MrCacheKind kind = it->second.kind;
    size_t budget = MrCacheBudgetBytes(kind);
    if (budget != 0 && MrCacheActiveBytes(kind) > budget) {
      while (MrCacheActiveBytes(kind) > budget) {
        if (!EvictOneCachedMRLocked("release", kind)) break;
      }
    }
    mr_cache_cv_.notify_all();
  }

  HeldMrKeys HoldRegisteredMR(MessageBuffer* msg_buf, char* ptr, size_t len,
                              uint64_t key, const char* label) {
    if (len == 0) return HeldMrKeys();
    std::lock_guard<std::mutex> lock(map_mu_);
    auto* mr = FindMRByRangeLocked(ptr, len);
    CHECK(mr) << "No MR for " << label << " ptr=" << (void*)ptr
              << ", len=" << len << ", key=" << key;
    HeldMrKeys keys;
    keys.lkey = mr->lkey;
    keys.rkey = mr->rkey;
    MRPtr ref = MakeCachedMRRefLocked(mr);
    msg_buf->mrs.push_back(std::make_pair(std::move(ref), len));
    return keys;
  }

  HeldMrKeys HoldOrRegisterMR(Message& msg, MessageBuffer* msg_buf, char* ptr,
                              size_t len, const char* label,
                              bool wait_for_budget = false) {
    if (len == 0) return HeldMrKeys();
    while (true) {
      {
        std::lock_guard<std::mutex> lock(map_mu_);
        auto* mr = FindMRByRangeLocked(ptr, len);
        if (mr != nullptr) {
          HeldMrKeys keys;
          keys.lkey = mr->lkey;
          keys.rkey = mr->rkey;
          MRPtr ref = MakeCachedMRRefLocked(mr);
          msg_buf->mrs.push_back(std::make_pair(std::move(ref), len));
          return keys;
        }
      }

      RegisterMemory(msg, wait_for_budget);

      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] retry hold after register key=%llu label=%s "
                "ptr=%p len=%zu\n",
                (unsigned long long)msg.meta.key, label, ptr, len);
        fflush(stderr);
      }
    }
  }

  HeldMrKeys KeysFromHeldMR(MessageBuffer* msg_buf, char* ptr, size_t len,
                            uint64_t key, const char* label) {
    CHECK(msg_buf);
    CHECK_EQ(msg_buf->mrs.size(), 1U)
        << "Missing pre-registered MR for " << label << ", key=" << key
        << ", ptr=" << (void*)ptr << ", len=" << len;
    auto* mr = msg_buf->mrs[0].first.get();
    CHECK(MrCoversRange(mr, ptr, len))
        << "Pre-registered MR does not cover " << label << ", key=" << key
        << ", ptr=" << (void*)ptr << ", len=" << len
        << ", mr.addr=" << mr->addr << ", mr.len=" << mr->length;
    HeldMrKeys keys;
    keys.lkey = mr->lkey;
    keys.rkey = mr->rkey;
    return keys;
  }

  void RepackMsgBufMeta(Message& msg, MessageBuffer* msg_buf) {
    int meta_len = GetPackMetaLen(msg.meta);
    CHECK_EQ(msg_buf->inline_len, (size_t)meta_len);
    PackMeta(msg.meta, &(msg_buf->inline_buf), &meta_len);
  }

  void PrepareRendezvousMR(Message& msg, MessageBuffer* msg_buf) {
    if (!IsValidPushpull(msg)) return;
    if (msg.meta.push && msg.meta.request) {
      msg_buf->mrs.clear();
      msg_buf->release_mrs_on_completion = true;
      PrepareData(msg, msg_buf, true);
    } else if (!msg.meta.push && msg.meta.request) {
      msg_buf->mrs.clear();
      msg_buf->release_mrs_on_completion = false;
      auto keys =
          HoldOrRegisterMR(msg, msg_buf, reinterpret_cast<char*>(msg.meta.addr),
                           msg.meta.val_len, "pull-request", true);
      msg.meta.option = keys.rkey;
      RepackMsgBufMeta(msg, msg_buf);
    } else if (!msg.meta.push && !msg.meta.request) {
      msg_buf->mrs.clear();
      msg_buf->release_mrs_on_completion = true;
      (void)HoldOrRegisterMR(msg, msg_buf, msg_buf->data[1].data(),
                             msg_buf->data[1].size(), "pull-response", true);
    } else {
      msg_buf->release_mrs_on_completion = true;
    }
  }

  void ReleasePullRequestMR(uint64_t key, int sender) {
    MessageBuffer* msg_buf = nullptr;
    {
      std::lock_guard<std::mutex> lk(addr_mu_);
      auto key_it = pull_addr_.find(key);
      if (key_it == pull_addr_.end()) return;
      auto peer_it = key_it->second.find(sender);
      if (peer_it == key_it->second.end()) return;
      msg_buf = std::get<3>(peer_it->second);
    }
    if (msg_buf) msg_buf->mrs.clear();
  }

  MrCacheKind ClassifyMRCacheKind(char* ptr, uint64_t key, const char* label) {
#if BYTEPS_RDMA_HAS_CUDA
    if (ptr == nullptr) return MrCacheKind::kHost;
    cudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    auto st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] kind attr failed key=%llu label=%s ptr=%p "
                "err=%s; use host budget\n",
                (unsigned long long)key, label, ptr, cudaGetErrorString(st));
        fflush(stderr);
      }
      cudaGetLastError();
      return MrCacheKind::kHost;
    }
#if CUDART_VERSION >= 10000
    auto mem_type = attr.type;
#else
    auto mem_type = attr.memoryType;
#endif
    bool is_device_like = (mem_type == cudaMemoryTypeDevice);
#if defined(cudaMemoryTypeManaged)
    is_device_like = is_device_like || (mem_type == cudaMemoryTypeManaged);
#endif
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] kind key=%llu label=%s ptr=%p mem_type=%d kind=%s\n",
              (unsigned long long)key, label, ptr, (int)mem_type,
              is_device_like ? "device" : "host");
      fflush(stderr);
    }
    return is_device_like ? MrCacheKind::kDevice : MrCacheKind::kHost;
#else
    (void)ptr;
    (void)key;
    (void)label;
    return MrCacheKind::kHost;
#endif
  }

  void RegisterMemory(Message& msg, bool wait_for_budget = false) {
    size_t sa_cnt = 0;
    for (auto& sa : msg.data) {
      if (sa.size() == 0) continue;
      std::unique_lock<std::mutex> lock(map_mu_);
      if (sa_cnt == 1) {  // only vals register memory
        // Push-request 的 vals 地址会随着 offset 变化。这里按“地址范围”查找，
        // 避免对同一块 GPU allocation 的每个分片重复 ibv_reg_mr。
        if (FindMRByRangeLocked(sa.data(), sa.size()) != nullptr) {
          ++sa_cnt;
          continue;
        }
        if (enable_mr_debug_) {
          auto p = reinterpret_cast<uintptr_t>(sa.data());
          auto pend = p + sa.size();
          for (auto& kv : mem_mr_) {
            auto* mr = kv.second;
            auto mr_base = reinterpret_cast<uintptr_t>(mr->addr);
            auto mr_end = mr_base + mr->length;
            if (p >= mr_base && p < mr_end && pend > mr_end) {
              fprintf(stderr,
                      "[MR-DEBUG] range miss (tail out) key=%llu ptr=%p "
                      "len=%zu mr.addr=%p mr.len=%u over=%zu\n",
                      (unsigned long long)msg.meta.key, sa.data(), sa.size(),
                      mr->addr, mr->length, pend - mr_end);
              fflush(stderr);
            }
          }
        }

        char* reg_ptr = sa.data();
        size_t reg_len = sa.size();
        RegisterRangeSource range_source = RegisterRangeSource::kOriginal;
        ExpandGpuRegisterRange(sa.data(), sa.size(), msg.meta.key, "reg-vals",
                               &reg_ptr, &reg_len,
                               &range_source);

        if (FindMRByRangeLocked(sa.data(), sa.size()) != nullptr) {
          ++sa_cnt;
          continue;
        }

        const int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        size_t try_len = reg_len;
        MrCacheKind mr_kind =
            (range_source == RegisterRangeSource::kCudaExact ||
             range_source == RegisterRangeSource::kProbeExpand)
                ? MrCacheKind::kDevice
                : ClassifyMRCacheKind(sa.data(), msg.meta.key, "reg-vals");
        EnsureCudaContextForPointer(reg_ptr, msg.meta.key, "reg-vals");
        struct ibv_mr* temp_mr = nullptr;
        bool exact_window_attempted = false;
#if BYTEPS_RDMA_HAS_CUDA && BYTEPS_RDMA_HAS_CUDA_DRIVER
        if (range_source == RegisterRangeSource::kCudaExact) {
          exact_window_attempted = true;
          (void)TryRegisterMrWithinCudaExactWindow(
              sa.data(), sa.size(), msg.meta.key, "vals-exact-local", mr_flags,
              &reg_ptr, &try_len, &temp_mr, &lock, wait_for_budget);
        }
#endif
        if (!temp_mr && !exact_window_attempted) {
          temp_mr = RegisterMRWithBudgetLocked(
              reg_ptr, try_len, mr_flags, msg.meta.key, "vals", mr_kind, &lock,
              wait_for_budget);
        }
        if (!temp_mr && !exact_window_attempted && errno == EFAULT) {
          (void)TryPrefaultCudaPages(reg_ptr, try_len, msg.meta.key, "vals");
          temp_mr = RegisterMRWithBudgetLocked(
              reg_ptr, try_len, mr_flags, msg.meta.key, "vals-prefault",
              mr_kind, &lock, wait_for_budget);
        }
        // Probe window can overestimate what NIC can register from this
        // address. Shrink registration size until success (but never below this
        // chunk).
        if (!temp_mr && errno == EFAULT && try_len > sa.size() &&
            range_source == RegisterRangeSource::kProbeExpand) {
          size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
          if (page == 0) page = 4096;
          size_t min_len = sa.size();
          while (!temp_mr && try_len > min_len) {
            size_t next = try_len >> 1;
            next = (next / page) * page;
            if (next < min_len) next = min_len;
            if (next == try_len) break;
            try_len = next;
            temp_mr = RegisterMRWithBudgetLocked(
                reg_ptr, try_len, mr_flags, msg.meta.key, "vals-shrink",
                mr_kind, &lock, wait_for_budget);
            if (!temp_mr && errno == EFAULT) {
              (void)TryPrefaultCudaPages(reg_ptr, try_len, msg.meta.key,
                                         "vals-shrink");
              temp_mr = RegisterMRWithBudgetLocked(
                  reg_ptr, try_len, mr_flags, msg.meta.key,
                  "vals-shrink-prefault", mr_kind, &lock, wait_for_budget);
            }
            if (enable_mr_debug_) {
              fprintf(
                  stderr,
                  "[MR-DEBUG] shrink reg window key=%llu ptr=%p try_len=%zu "
                  "result=%s\n",
                  (unsigned long long)msg.meta.key, reg_ptr, try_len,
                  temp_mr ? "ok" : "fail");
              fflush(stderr);
            }
          }
        }
        // Some GPU virtual addresses fail to register when used as MR base,
        // while a slightly earlier base that still covers [sa, sa+len) works.
        if (!temp_mr && errno == EFAULT && enable_reg_backoff_ &&
            range_source != RegisterRangeSource::kCudaExact) {
          size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
          if (page == 0) page = 4096;
          uintptr_t p = reinterpret_cast<uintptr_t>(reg_ptr);
          size_t max_back = 8UL * 1024UL * 1024UL;  // up to 8MB backoff
          for (size_t back = page; back <= max_back; back += page) {
            if (p < back) break;
            char* cand_ptr = reinterpret_cast<char*>(p - back);
            size_t cand_len = sa.size() + back;
            struct ibv_mr* cand_mr = RegisterMRWithBudgetLocked(
                cand_ptr, cand_len, mr_flags, msg.meta.key, "vals-backoff",
                mr_kind, &lock, wait_for_budget);
            if (!cand_mr && errno == EFAULT) {
              (void)TryPrefaultCudaPages(cand_ptr, cand_len, msg.meta.key,
                                         "vals-backoff");
              cand_mr = RegisterMRWithBudgetLocked(
                  cand_ptr, cand_len, mr_flags, msg.meta.key,
                  "vals-backoff-prefault", mr_kind, &lock, wait_for_budget);
            }
            if (enable_mr_debug_) {
              fprintf(stderr,
                      "[MR-DEBUG] backoff reg window key=%llu cand.ptr=%p "
                      "cand.len=%zu result=%s\n",
                      (unsigned long long)msg.meta.key, cand_ptr, cand_len,
                      cand_mr ? "ok" : "fail");
              fflush(stderr);
            }
            if (cand_mr) {
              temp_mr = cand_mr;
              reg_ptr = cand_ptr;
              try_len = cand_len;
              break;
            }
          }
        }
        CHECK(temp_mr) << "Failed to register the memory region: "
                       << strerror(errno) << ", key=" << msg.meta.key
                       << ", sa.ptr=" << (void*)sa.data()
                       << ", sa.size()=" << sa.size()
                       << ", reg.ptr=" << (void*)reg_ptr
                       << ", reg.size=" << try_len
                       << ", reg.source="
                       << RegisterRangeSourceName(range_source);
        reg_len = try_len;
        PS_VLOG(1) << "Register Mem .Sa Size" << sa.size();
        if (enable_mr_debug_) {
          fprintf(stderr,
                  "[MR-DEBUG] reg_mr key=%llu ptr=%p len=%zu reg.ptr=%p "
                  "reg.len=%zu lkey=%u rkey=%u\n",
                  (unsigned long long)msg.meta.key, sa.data(), sa.size(),
                  reg_ptr, reg_len, temp_mr->lkey, temp_mr->rkey);
          fflush(stderr);
        }
        // 用注册基址作为 key，避免同一 MR 被重复插入后在析构时重复 dereg。
        InsertCachedMRLocked(temp_mr, msg.meta.key, "vals", mr_kind);
      }
      ++sa_cnt;
    }
    // register for tensor address of pull request
    if (IsValidPushpull(msg) && !msg.meta.push && msg.meta.request) {
      CHECK_GT(msg.meta.val_len, 0) << msg.meta.val_len;
      auto addr = reinterpret_cast<char*>(msg.meta.addr);
      std::unique_lock<std::mutex> lock(map_mu_);
      if (FindMRByRangeLocked(addr, msg.meta.val_len) == nullptr) {
        char* reg_ptr = addr;
        size_t reg_len = msg.meta.val_len;
        RegisterRangeSource range_source = RegisterRangeSource::kOriginal;
        ExpandGpuRegisterRange(addr, msg.meta.val_len, msg.meta.key,
                               "reg-pull", &reg_ptr, &reg_len,
                               &range_source);

        if (FindMRByRangeLocked(addr, msg.meta.val_len) != nullptr) {
          return;
        }

        const int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        size_t try_len = reg_len;
        MrCacheKind mr_kind =
            (range_source == RegisterRangeSource::kCudaExact ||
             range_source == RegisterRangeSource::kProbeExpand)
                ? MrCacheKind::kDevice
                : ClassifyMRCacheKind(addr, msg.meta.key, "reg-pull");
        EnsureCudaContextForPointer(reg_ptr, msg.meta.key, "reg-pull");
        struct ibv_mr* temp_mr = nullptr;
        bool exact_window_attempted = false;
#if BYTEPS_RDMA_HAS_CUDA && BYTEPS_RDMA_HAS_CUDA_DRIVER
        if (range_source == RegisterRangeSource::kCudaExact) {
          exact_window_attempted = true;
          (void)TryRegisterMrWithinCudaExactWindow(
              addr, msg.meta.val_len, msg.meta.key, "pull-exact-local",
              mr_flags, &reg_ptr, &try_len, &temp_mr, &lock, wait_for_budget);
        }
#endif
        if (!temp_mr && !exact_window_attempted) {
          temp_mr = RegisterMRWithBudgetLocked(
              reg_ptr, try_len, mr_flags, msg.meta.key, "pull", mr_kind, &lock,
              wait_for_budget);
        }
        if (!temp_mr && !exact_window_attempted && errno == EFAULT) {
          (void)TryPrefaultCudaPages(reg_ptr, try_len, msg.meta.key, "pull");
          temp_mr = RegisterMRWithBudgetLocked(
              reg_ptr, try_len, mr_flags, msg.meta.key, "pull-prefault",
              mr_kind, &lock, wait_for_budget);
        }
        if (!temp_mr && errno == EFAULT &&
            try_len > (size_t)msg.meta.val_len &&
            range_source == RegisterRangeSource::kProbeExpand) {
          size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
          if (page == 0) page = 4096;
          size_t min_len = msg.meta.val_len;
          while (!temp_mr && try_len > min_len) {
            size_t next = try_len >> 1;
            next = (next / page) * page;
            if (next < min_len) next = min_len;
            if (next == try_len) break;
            try_len = next;
            temp_mr = RegisterMRWithBudgetLocked(
                reg_ptr, try_len, mr_flags, msg.meta.key, "pull-shrink",
                mr_kind, &lock, wait_for_budget);
            if (!temp_mr && errno == EFAULT) {
              (void)TryPrefaultCudaPages(reg_ptr, try_len, msg.meta.key,
                                         "pull-shrink");
              temp_mr = RegisterMRWithBudgetLocked(
                  reg_ptr, try_len, mr_flags, msg.meta.key,
                  "pull-shrink-prefault", mr_kind, &lock, wait_for_budget);
            }
            if (enable_mr_debug_) {
              fprintf(stderr,
                      "[MR-DEBUG] shrink pull reg window key=%llu ptr=%p "
                      "try_len=%zu result=%s\n",
                      (unsigned long long)msg.meta.key, reg_ptr, try_len,
                      temp_mr ? "ok" : "fail");
              fflush(stderr);
            }
          }
        }
        // Pull destination buffers can also be partitioned GPU addresses, so
        // mirror the push-path backoff logic here instead of failing on the
        // first interior-address registration error.
        if (!temp_mr && errno == EFAULT && enable_reg_backoff_ &&
            range_source != RegisterRangeSource::kCudaExact) {
          size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
          if (page == 0) page = 4096;
          uintptr_t p = reinterpret_cast<uintptr_t>(reg_ptr);
          size_t max_back = 8UL * 1024UL * 1024UL;  // up to 8MB backoff
          for (size_t back = page; back <= max_back; back += page) {
            if (p < back) break;
            char* cand_ptr = reinterpret_cast<char*>(p - back);
            size_t cand_len = msg.meta.val_len + back;
            struct ibv_mr* cand_mr = RegisterMRWithBudgetLocked(
                cand_ptr, cand_len, mr_flags, msg.meta.key, "pull-backoff",
                mr_kind, &lock, wait_for_budget);
            if (!cand_mr && errno == EFAULT) {
              (void)TryPrefaultCudaPages(cand_ptr, cand_len, msg.meta.key,
                                         "pull-backoff");
              cand_mr = RegisterMRWithBudgetLocked(
                  cand_ptr, cand_len, mr_flags, msg.meta.key,
                  "pull-backoff-prefault", mr_kind, &lock, wait_for_budget);
            }
            if (enable_mr_debug_) {
              fprintf(stderr,
                      "[MR-DEBUG] backoff pull reg window key=%llu "
                      "cand.ptr=%p cand.len=%zu result=%s\n",
                      (unsigned long long)msg.meta.key, cand_ptr, cand_len,
                      cand_mr ? "ok" : "fail");
              fflush(stderr);
            }
            if (cand_mr) {
              temp_mr = cand_mr;
              reg_ptr = cand_ptr;
              try_len = cand_len;
              break;
            }
          }
        }
        CHECK(temp_mr) << "Failed to register the memory region: "
                       << strerror(errno) << ", key=" << msg.meta.key
                       << ", addr=" << (void*)addr
                       << ", len=" << msg.meta.val_len
                       << ", reg.ptr=" << (void*)reg_ptr
                       << ", reg.size=" << try_len
                       << ", reg.source="
                       << RegisterRangeSourceName(range_source);
        reg_len = try_len;
        PS_VLOG(1) << "Register Mem .val_len " << msg.meta.val_len;
        if (enable_mr_debug_) {
          fprintf(stderr,
                  "[MR-DEBUG] reg_mr pull key=%llu ptr=%p len=%u reg.ptr=%p "
                  "reg.len=%zu lkey=%u rkey=%u\n",
                  (unsigned long long)msg.meta.key, addr, msg.meta.val_len,
                  reg_ptr, reg_len, temp_mr->lkey, temp_mr->rkey);
          fflush(stderr);
        }
        InsertCachedMRLocked(temp_mr, msg.meta.key, "pull", mr_kind);
      }
    }
  }

  bool TryPrefaultCudaPages(void* ptr, size_t len, uint64_t key,
                            const char* label) {
#if BYTEPS_RDMA_HAS_CUDA
    if (ptr == nullptr || len == 0) return false;
    EnsureCudaContextForPointer(reinterpret_cast<char*>(ptr), key, label);
    cudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    auto st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] prefault attr failed key=%llu label=%s ptr=%p "
                "len=%zu err=%s\n",
                (unsigned long long)key, label, ptr, len,
                cudaGetErrorString(st));
        fflush(stderr);
      }
      cudaGetLastError();
      return false;
    }
#if CUDART_VERSION >= 10000
    auto mem_type = attr.type;
#else
    auto mem_type = attr.memoryType;
#endif
    bool is_device_like = (mem_type == cudaMemoryTypeDevice);
#if defined(cudaMemoryTypeManaged)
    is_device_like = is_device_like || (mem_type == cudaMemoryTypeManaged);
#endif
    if (!is_device_like) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] prefault skip non-device key=%llu label=%s ptr=%p "
                "mem_type=%d\n",
                (unsigned long long)key, label, ptr, (int)mem_type);
        fflush(stderr);
      }
      return false;
    }

    const size_t step = 64 * 1024;
    volatile unsigned char sink = 0;
    auto* p = reinterpret_cast<const char*>(ptr);
    bool ok = true;
    for (size_t off = 0; off < len; off += step) {
      unsigned char v = 0;
      auto s = cudaMemcpy(&v, p + off, 1, cudaMemcpyDeviceToHost);
      if (s != cudaSuccess) {
        ok = false;
        if (enable_mr_debug_) {
          fprintf(stderr,
                  "[MR-DEBUG] prefault fail key=%llu label=%s ptr=%p off=%zu "
                  "len=%zu err=%s\n",
                  (unsigned long long)key, label, ptr, off, len,
                  cudaGetErrorString(s));
          fflush(stderr);
        }
        cudaGetLastError();
        break;
      }
      sink ^= v;
    }
    if (ok && len > 1) {
      unsigned char v = 0;
      auto s = cudaMemcpy(&v, p + (len - 1), 1, cudaMemcpyDeviceToHost);
      if (s != cudaSuccess) {
        ok = false;
        if (enable_mr_debug_) {
          fprintf(stderr,
                  "[MR-DEBUG] prefault tail fail key=%llu label=%s ptr=%p "
                  "len=%zu err=%s\n",
                  (unsigned long long)key, label, ptr, len,
                  cudaGetErrorString(s));
          fflush(stderr);
        }
        cudaGetLastError();
      } else {
        sink ^= v;
      }
    }
    (void)sink;
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] prefault %s key=%llu ptr=%p len=%zu result=%s\n",
              label, (unsigned long long)key, ptr, len, ok ? "ok" : "fail");
      fflush(stderr);
    }
    return ok;
#else
    (void)ptr;
    (void)len;
    (void)key;
    (void)label;
    return false;
#endif
  }

  const char* RegisterRangeSourceName(RegisterRangeSource source) const {
    switch (source) {
      case RegisterRangeSource::kOriginal:
        return "original";
      case RegisterRangeSource::kCudaExact:
        return "cuda-exact";
      case RegisterRangeSource::kProbeExpand:
        return "probe-expand";
    }
    return "unknown";
  }

#if BYTEPS_RDMA_HAS_CUDA && BYTEPS_RDMA_HAS_CUDA_DRIVER
  struct CudaAllocRange {
    char* base = nullptr;
    size_t size = 0;
    size_t offset = 0;
  };

  struct CudaDriverApi {
    using CuInitFn = CUresult(CUDAAPI*)(unsigned int);
    using CuPointerGetAttributeFn =
        CUresult(CUDAAPI*)(void*, CUpointer_attribute, CUdeviceptr);
    using CuMemGetAddressRangeFn =
        CUresult(CUDAAPI*)(CUdeviceptr*, size_t*, CUdeviceptr);

    void* handle = nullptr;
    CuInitFn cuInit = nullptr;
    CuPointerGetAttributeFn cuPointerGetAttribute = nullptr;
    CuMemGetAddressRangeFn cuMemGetAddressRange = nullptr;
    bool ready = false;
  };

  CudaDriverApi& GetCudaDriverApi() {
    static CudaDriverApi api = []() {
      CudaDriverApi driver_api;
      driver_api.cuInit = reinterpret_cast<CudaDriverApi::CuInitFn>(
          dlsym(RTLD_DEFAULT, "cuInit"));
      driver_api.cuPointerGetAttribute =
          reinterpret_cast<CudaDriverApi::CuPointerGetAttributeFn>(
              dlsym(RTLD_DEFAULT, "cuPointerGetAttribute"));
      driver_api.cuMemGetAddressRange =
          reinterpret_cast<CudaDriverApi::CuMemGetAddressRangeFn>(
              dlsym(RTLD_DEFAULT, "cuMemGetAddressRange"));
      driver_api.ready =
          driver_api.cuInit != nullptr &&
          driver_api.cuPointerGetAttribute != nullptr;
      if (driver_api.ready) {
        return driver_api;
      }

      const char* candidates[] = {"libcuda.so.1", "libcuda.so"};
      for (const char* candidate : candidates) {
        driver_api.handle = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
        if (driver_api.handle != nullptr) break;
      }
      if (driver_api.handle == nullptr) {
        return driver_api;
      }

      driver_api.cuInit = reinterpret_cast<CudaDriverApi::CuInitFn>(
          dlsym(driver_api.handle, "cuInit"));
      driver_api.cuPointerGetAttribute =
          reinterpret_cast<CudaDriverApi::CuPointerGetAttributeFn>(
              dlsym(driver_api.handle, "cuPointerGetAttribute"));
      driver_api.cuMemGetAddressRange =
          reinterpret_cast<CudaDriverApi::CuMemGetAddressRangeFn>(
              dlsym(driver_api.handle, "cuMemGetAddressRange"));
      driver_api.ready =
          driver_api.cuInit != nullptr &&
          driver_api.cuPointerGetAttribute != nullptr;
      return driver_api;
    }();
    return api;
  }

  static uintptr_t AlignDownPow2(uintptr_t value, size_t align) {
    return value & ~(static_cast<uintptr_t>(align) - 1);
  }

  static uintptr_t AlignUpPow2(uintptr_t value, size_t align) {
    return (value + static_cast<uintptr_t>(align) - 1) &
           ~(static_cast<uintptr_t>(align) - 1);
  }

  bool TryGetExactCudaAllocRange(char* ptr, size_t len, uint64_t key,
                                 const char* label, CudaAllocRange* range) {
    if (ptr == nullptr || len == 0) return false;
    CHECK(range);

    EnsureCudaContextForPointer(ptr, key, label);

    cudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    auto st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] exact range attr failed key=%llu label=%s ptr=%p "
                "len=%zu err=%s\n",
                (unsigned long long)key, label, ptr, len,
                cudaGetErrorString(st));
        fflush(stderr);
      }
      cudaGetLastError();
      return false;
    }

#if CUDART_VERSION >= 10000
    auto mem_type = attr.type;
#else
    auto mem_type = attr.memoryType;
#endif
    bool is_device_like = (mem_type == cudaMemoryTypeDevice);
#if defined(cudaMemoryTypeManaged)
    is_device_like = is_device_like || (mem_type == cudaMemoryTypeManaged);
#endif
    if (!is_device_like) {
      return false;
    }

    auto& driver_api = GetCudaDriverApi();
    if (!driver_api.ready) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] exact range unavailable key=%llu label=%s ptr=%p "
                "len=%zu reason=driver-api-not-ready\n",
                (unsigned long long)key, label, ptr, len);
        fflush(stderr);
      }
      return false;
    }

    auto init_rc = driver_api.cuInit(0);
    if (init_rc != CUDA_SUCCESS) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] exact range init failed key=%llu label=%s ptr=%p "
                "len=%zu cuInit=%d\n",
                (unsigned long long)key, label, ptr, len, (int)init_rc);
        fflush(stderr);
      }
      return false;
    }

    CUdeviceptr query = reinterpret_cast<CUdeviceptr>(ptr);
    CUdeviceptr base = 0;
    size_t range_size = 0;
    auto base_rc = driver_api.cuPointerGetAttribute(
        &base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, query);
    auto size_rc = driver_api.cuPointerGetAttribute(
        &range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, query);
    if (base_rc != CUDA_SUCCESS || size_rc != CUDA_SUCCESS || base == 0 ||
        range_size == 0) {
      if (driver_api.cuMemGetAddressRange != nullptr) {
        auto mem_rc =
            driver_api.cuMemGetAddressRange(&base, &range_size, query);
        if (mem_rc != CUDA_SUCCESS || base == 0 || range_size == 0) {
          if (enable_mr_debug_) {
            fprintf(stderr,
                    "[MR-DEBUG] exact range failed key=%llu label=%s ptr=%p "
                    "len=%zu base_rc=%d size_rc=%d mem_rc=%d\n",
                    (unsigned long long)key, label, ptr, len, (int)base_rc,
                    (int)size_rc, (int)mem_rc);
            fflush(stderr);
          }
          return false;
        }
      } else {
        if (enable_mr_debug_) {
          fprintf(stderr,
                  "[MR-DEBUG] exact range failed key=%llu label=%s ptr=%p "
                  "len=%zu base_rc=%d size_rc=%d\n",
                  (unsigned long long)key, label, ptr, len, (int)base_rc,
                  (int)size_rc);
          fflush(stderr);
        }
        return false;
      }
    }

    auto p = reinterpret_cast<uintptr_t>(ptr);
    auto b = static_cast<uintptr_t>(base);
    if (p < b) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] exact range invalid key=%llu label=%s ptr=%p "
                "base=%p len=%zu range_size=%zu reason=ptr-before-base\n",
                (unsigned long long)key, label, ptr, (void*)b, len,
                range_size);
        fflush(stderr);
      }
      return false;
    }

    size_t offset = p - b;
    if (offset > range_size || len > range_size - offset) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] exact range insufficient key=%llu label=%s ptr=%p "
                "len=%zu base=%p range_size=%zu offset=%zu\n",
                (unsigned long long)key, label, ptr, len, (void*)b, range_size,
                offset);
        fflush(stderr);
      }
      return false;
    }

    range->base = reinterpret_cast<char*>(b);
    range->size = range_size;
    range->offset = offset;
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] exact range key=%llu label=%s ptr=%p len=%zu "
              "base=%p range.size=%zu offset=%zu\n",
              (unsigned long long)key, label, ptr, len, range->base,
              range->size, range->offset);
      fflush(stderr);
    }
    return true;
  }

  bool TryRegisterMrWithinCudaExactWindow(char* ptr, size_t len, uint64_t key,
                                          const char* label, int mr_flags,
                                          char** reg_ptr, size_t* reg_len,
                                          struct ibv_mr** temp_mr,
                                          std::unique_lock<std::mutex>* lock,
                                          bool wait_for_budget) {
    if (ptr == nullptr || len == 0) return false;
    CHECK(reg_ptr);
    CHECK(reg_len);
    CHECK(temp_mr);

    CudaAllocRange range;
    if (!TryGetExactCudaAllocRange(ptr, len, key, label, &range)) {
      return false;
    }

    size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (page == 0) page = 4096;
    size_t exact_local_cache = 4UL * 1024UL * 1024UL;
    auto exact_local_cache_env =
        Environment::Get()->find("BYTEPS_RDMA_EXACT_LOCAL_CACHE_MB");
    if (exact_local_cache_env) {
      long v = atol(exact_local_cache_env);
      if (v > 0) {
        exact_local_cache = static_cast<size_t>(v) * 1024UL * 1024UL;
      }
    }
    if (exact_local_cache < page) exact_local_cache = page;
    if (exact_local_cache & (exact_local_cache - 1)) {
      size_t rounded = 1;
      while (rounded < exact_local_cache) rounded <<= 1;
      exact_local_cache = rounded;
    }
    auto p = reinterpret_cast<uintptr_t>(ptr);
    auto req_end = p + len;
    auto base = reinterpret_cast<uintptr_t>(range.base);
    auto alloc_end = base + range.size;

    uintptr_t prev_start = std::numeric_limits<uintptr_t>::max();
    uintptr_t prev_end = std::numeric_limits<uintptr_t>::max();
    auto try_window = [&](uintptr_t cand_start, uintptr_t cand_end,
                          const char* mode, size_t param) {
      if (cand_start < base) cand_start = base;
      if (cand_end > alloc_end) cand_end = alloc_end;
      cand_start = AlignDownPow2(cand_start, page);
      cand_end = AlignUpPow2(cand_end, page);
      if (cand_start < base) cand_start = base;
      if (cand_end > alloc_end) cand_end = alloc_end;
      if (cand_start > p || cand_end < req_end || cand_end <= cand_start) {
        return false;
      }
      if (cand_start == prev_start && cand_end == prev_end) {
        return false;
      }
      prev_start = cand_start;
      prev_end = cand_end;

      char* cand_ptr = reinterpret_cast<char*>(cand_start);
      size_t cand_len = cand_end - cand_start;
      *reg_ptr = cand_ptr;
      *reg_len = cand_len;
      auto* cand_mr = RegisterMRWithBudgetLocked(
          cand_ptr, cand_len, mr_flags, key, label, MrCacheKind::kDevice, lock,
          wait_for_budget);
      if (!cand_mr && errno == EFAULT) {
        (void)TryPrefaultCudaPages(cand_ptr, cand_len, key, label);
        cand_mr = RegisterMRWithBudgetLocked(
            cand_ptr, cand_len, mr_flags, key, label, MrCacheKind::kDevice,
            lock, wait_for_budget);
      }
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] exact local reg key=%llu label=%s ptr=%p len=%zu "
                "cand.ptr=%p cand.len=%zu mode=%s param=%zu result=%s\n",
                (unsigned long long)key, label, ptr, len, cand_ptr, cand_len,
                mode, param, cand_mr ? "ok" : "fail");
        fflush(stderr);
      }
      if (cand_mr) {
        *temp_mr = cand_mr;
        return true;
      }
      return false;
    };

    // CUDA exact allocations are already known. Keep registration candidates
    // small and non-overlapping so the production path matches gdr_probe's
    // successful cache-window model. Expanding to 8/16/64MiB or to the
    // allocation prefix can overlap existing MRs and amplify a transient EFAULT
    // into a much larger Bad address failure.
    if (try_window(AlignDownPow2(p, exact_local_cache),
                   AlignUpPow2(req_end, exact_local_cache),
                   "exact-cache-window", exact_local_cache)) {
      return true;
    }

    if (try_window(p, req_end, "exact-request", len)) {
      return true;
    }

    // Some CUDA VAs still fail when the MR starts exactly at the requested
    // slice, even though a slightly earlier MR base works. Keep this fallback
    // bounded inside the exact CUDA allocation and inside the local cache
    // window, so it cannot regress to the old huge exact-prefix registration.
    size_t max_back = 0;
    if (exact_local_cache > len) {
      max_back = exact_local_cache - len;
    }
    size_t max_alloc_back = p > base ? p - base : 0;
    if (max_back > max_alloc_back) {
      max_back = max_alloc_back;
    }
    size_t last_back = 0;
    for (size_t back = page; back <= max_back && back > last_back;) {
      if (try_window(p - back, req_end, "exact-backoff-tail", back)) {
        return true;
      }
      last_back = back;
      size_t next = back << 1;
      if (next <= back || next > max_back) {
        next = max_back;
      }
      back = next;
    }

    return false;
  }
#endif

  void EnsureCudaContextForPointer(char* ptr, uint64_t key, const char* label) {
#if BYTEPS_RDMA_HAS_CUDA
    if (ptr == nullptr) return;
    // Ensure this thread has a CUDA context before querying pointer attrs.
    thread_local bool thread_ctx_inited = false;
    if (!thread_ctx_inited) {
      int init_dev = 0;
      auto* lr = Environment::Get()->find("BYTEPS_LOCAL_RANK");
      if (lr) {
        init_dev = atoi(lr);
        if (init_dev < 0) init_dev = 0;
      }
      auto s0 = cudaSetDevice(init_dev);
      if (s0 == cudaSuccess) {
        (void)cudaFree(0);
        thread_ctx_inited = true;
      } else {
        cudaGetLastError();
      }
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] ctx init key=%llu label=%s init_dev=%d result=%s\n",
                (unsigned long long)key, label, init_dev,
                s0 == cudaSuccess ? "ok" : "fail");
        fflush(stderr);
      }
    }

    cudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    auto st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] ctx attr failed key=%llu label=%s ptr=%p err=%s\n",
                (unsigned long long)key, label, ptr, cudaGetErrorString(st));
        fflush(stderr);
      }
      cudaGetLastError();
      return;
    }
#if CUDART_VERSION >= 10000
    auto mem_type = attr.type;
#else
    auto mem_type = attr.memoryType;
#endif
    bool is_device_like = (mem_type == cudaMemoryTypeDevice);
#if defined(cudaMemoryTypeManaged)
    is_device_like = is_device_like || (mem_type == cudaMemoryTypeManaged);
#endif
    if (!is_device_like) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] ctx skip non-device key=%llu label=%s ptr=%p "
                "mem_type=%d\n",
                (unsigned long long)key, label, ptr, (int)mem_type);
        fflush(stderr);
      }
      return;
    }

    int cur_dev = -1;
    auto st_cur = cudaGetDevice(&cur_dev);
    if (st_cur != cudaSuccess) {
      cudaGetLastError();
      cur_dev = -1;
    }
    int target_dev = attr.device;
    if (target_dev >= 0 && cur_dev != target_dev) {
      auto s1 = cudaSetDevice(target_dev);
      auto s2 = (s1 == cudaSuccess) ? cudaFree(0) : cudaErrorUnknown;
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] ctx align key=%llu label=%s ptr=%p cur_dev=%d "
                "target_dev=%d set=%s free0=%s\n",
                (unsigned long long)key, label, ptr, cur_dev, target_dev,
                s1 == cudaSuccess ? "ok" : "fail",
                s2 == cudaSuccess ? "ok" : "fail");
        fflush(stderr);
      }
      if (s1 != cudaSuccess || s2 != cudaSuccess) {
        cudaGetLastError();
      }
    }
#else
    (void)ptr;
    (void)key;
    (void)label;
#endif
  }

  struct ibv_mr* FindMRByRangeLocked(char* ptr, size_t len) {
    auto direct_it = mem_mr_.find(ptr);
    if (direct_it != mem_mr_.end()) {
      auto* mr = direct_it->second;
      auto mr_base = reinterpret_cast<uintptr_t>(mr->addr);
      auto mr_end = mr_base + mr->length;
      auto p = reinterpret_cast<uintptr_t>(ptr);
      auto pend = p + len;
      if (p >= mr_base && pend <= mr_end) {
        TouchCachedMRLocked(mr);
        return mr;
      }
    }

    auto p = reinterpret_cast<uintptr_t>(ptr);
    auto pend = p + len;
    for (auto& kv : mem_mr_) {
      auto* mr = kv.second;
      auto mr_base = reinterpret_cast<uintptr_t>(mr->addr);
      auto mr_end = mr_base + mr->length;
      if (p >= mr_base && pend <= mr_end) {
        TouchCachedMRLocked(mr);
        return mr;
      }
    }
    return nullptr;
  }

  void ExpandGpuRegisterRange(char* ptr, size_t len, uint64_t key,
                              const char* label, char** reg_ptr,
                              size_t* reg_len,
                              RegisterRangeSource* range_source) {
    *reg_ptr = ptr;
    *reg_len = len;
    if (range_source) {
      *range_source = RegisterRangeSource::kOriginal;
    }

#if BYTEPS_RDMA_HAS_CUDA && BYTEPS_RDMA_HAS_CUDA_DRIVER
    CudaAllocRange range;
    if (TryGetExactCudaAllocRange(ptr, len, key, label, &range)) {
      *reg_ptr = range.base;
      *reg_len = range.offset + len;
      if (range_source) {
        *range_source = RegisterRangeSource::kCudaExact;
      }
      return;
    }
#endif

    // Allow disabling only the probe-based heuristic expansion.
    // Exact CUDA allocation range discovery above remains enabled because it
    // avoids guessing across allocation boundaries.
    auto no_expand_env = Environment::Get()->find("BYTEPS_RDMA_NO_EXPAND");
    if (no_expand_env && atoi(no_expand_env) != 0) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] expand disabled by BYTEPS_RDMA_NO_EXPAND: ptr=%p "
                "len=%zu\n",
                ptr, len);
        fflush(stderr);
      }
      return;
    }

#if BYTEPS_RDMA_HAS_CUDA
    if (ptr == nullptr || len == 0) return;
    cudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    auto st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess) {
      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] expand attr failed: ptr=%p len=%zu err=%s\n", ptr,
                len, cudaGetErrorString(st));
        fflush(stderr);
      }
      cudaGetLastError();
      return;
    }
#if CUDART_VERSION >= 10000
    auto mem_type = attr.type;
#else
    auto mem_type = attr.memoryType;
#endif
    bool is_device_like = (mem_type == cudaMemoryTypeDevice);
#if defined(cudaMemoryTypeManaged)
    is_device_like = is_device_like || (mem_type == cudaMemoryTypeManaged);
#endif
    if (!is_device_like) {
      if (enable_mr_debug_) {
        fprintf(
            stderr,
            "[MR-DEBUG] expand skip non-device: ptr=%p len=%zu mem_type=%d\n",
            ptr, len, (int)mem_type);
        fflush(stderr);
      }
      return;
    }

    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] expand range by probe only: ptr=%p len=%zu "
              "(cudaMemGetAddressRange unavailable in this CUDA runtime)\n",
              ptr, len);
      fflush(stderr);
    }

    // Fallback: probe a larger readable range from ptr and register a window.
    auto is_readable = [&](size_t n) -> bool {
      if (n == 0) return false;
      unsigned char v = 0;
      auto s = cudaMemcpy(&v, ptr + (n - 1), 1, cudaMemcpyDeviceToHost);
      if (s != cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      return true;
    };

    size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (page == 0) page = 4096;
    size_t max_window = 64UL * 1024UL * 1024UL;
    auto env_window_mb = Environment::Get()->find("BYTEPS_RDMA_REG_WINDOW_MB");
    if (env_window_mb) {
      long v = atol(env_window_mb);
      if (v > 0) max_window = static_cast<size_t>(v) * 1024UL * 1024UL;
    }
    if (max_window < len) max_window = len;

    size_t lo = len;
    size_t hi = len;
    while (hi < max_window && is_readable(hi)) {
      lo = hi;
      size_t next = hi << 1;
      hi = (next > max_window) ? max_window : next;
      if (hi == lo) break;
    }

    if (!is_readable(hi)) {
      size_t l = lo;
      size_t r = hi;
      while (l + page < r) {
        size_t mid = l + ((r - l) / 2);
        mid = (mid / page) * page;
        if (mid <= l) mid = l + page;
        if (is_readable(mid)) {
          l = mid;
        } else {
          r = mid;
        }
      }
      lo = l;
    } else {
      lo = hi;
    }

    if (lo < len) lo = len;
    lo = (lo / page) * page;
    if (lo < len) lo = len;
    *reg_ptr = ptr;
    *reg_len = lo;
    if (range_source && *reg_len > len) {
      *range_source = RegisterRangeSource::kProbeExpand;
    }
    if (enable_mr_debug_) {
      fprintf(stderr,
              "[MR-DEBUG] expand range by probe: ptr=%p len=%zu -> reg.len=%zu "
              "(max_window=%zu)\n",
              ptr, len, *reg_len, max_window);
      fflush(stderr);
    }
#else
    (void)ptr;
    (void)len;
#endif
  }

  void PrepareData(Message& msg, MessageBuffer* msg_buf,
                   bool wait_for_budget = false) {
    if (!(msg.meta.push && msg.meta.request)) return;  // only push request
    auto& sa = msg_buf->data[1];
    if (sa.size() == 0) return;
    while (true) {
      {
        std::lock_guard<std::mutex> lock(map_mu_);
        auto* mr = FindMRByRangeLocked(sa.data(), sa.size());
        if (mr != nullptr) {
          MRPtr ptr = MakeCachedMRRefLocked(mr);
          msg_buf->mrs.push_back(std::make_pair(std::move(ptr), sa.size()));
          return;
        }
      }

      // RegisterMemory() drops map_mu_ before this function can take an MR
      // reference. Another thread may evict that unreferenced MR immediately,
      // so retry until we atomically find and hold a covering MR.
      RegisterMemory(msg, wait_for_budget);

      if (enable_mr_debug_) {
        fprintf(stderr,
                "[MR-DEBUG] retry push hold after register key=%llu "
                "ptr=%p len=%zu\n",
                (unsigned long long)msg.meta.key, sa.data(), sa.size());
        fflush(stderr);
      }
    }
  }

  void AddMeta(Message& msg) {
    if (msg.meta.request) {
      msg.meta.key = DecodeKey(msg.data[0]);
    }
  }

  void InitContext(struct ibv_context* context) {
    context_ = context;
    CHECK(context_) << "ibv_context* empty";

    pd_ = ibv_alloc_pd(context_);
    CHECK(pd_) << "Failed to allocate protection domain";

    mem_allocator_.reset(new MemoryAllocator(pd_));

    comp_event_channel_ = ibv_create_comp_channel(context_);

    // TODO(clan): Replace the rough estimate here
    cq_ = ibv_create_cq(context_, kMaxConcurrentWorkRequest * 2, NULL,
                        comp_event_channel_, 0);
    auto use_srq_env = Environment::Get()->find("BYTEPS_RDMA_USE_SRQ");
    use_srq_ = use_srq_env ? atoi(use_srq_env) : true;
    if (use_srq_) {
      auto srq_depth_env = Environment::Get()->find("BYTEPS_RDMA_SRQ_DEPTH");
      srq_depth_ = srq_depth_env ? atoi(srq_depth_env) : 4096;

      struct ibv_srq_init_attr srq_attr;
      memset(&srq_attr, 0, sizeof(srq_attr));
      srq_attr.attr.max_wr = srq_depth_;
      srq_attr.attr.max_sge = kSGEntry;
      srq_ = ibv_create_srq(pd_, &srq_attr);
      CHECK(srq_) << "Failed to create SRQ";

      shared_rx_ctx_ = new WRContext[srq_depth_];
      for (int i = 0; i < srq_depth_; ++i) {
        void* buf;
        aligned_malloc((void**)&buf, kMempoolChunkSize);
        CHECK(buf);
        struct ibv_mr* mr =
            ibv_reg_mr(pd_, buf, kMempoolChunkSize, IBV_ACCESS_LOCAL_WRITE);
        CHECK(mr) << "ibv_reg_mr failed for SRQ: " << strerror(errno);

        shared_rx_ctx_[i].type = kReceiveContext;
        shared_rx_ctx_[i].buffer = mr;
        shared_rx_ctx_[i].private_data = nullptr;
        PostSharedRecv(&shared_rx_ctx_[i]);
      }
    }

    CHECK(cq_) << "Failed to create completion queue";
    CHECK(!ibv_req_notify_cq(cq_, 0)) << "Failed to request CQ notification";
  }

  void PostSharedRecv(WRContext* ctx) {
    struct ibv_recv_wr wr, *bad_wr = nullptr;
    memset(&wr, 0, sizeof(wr));

    struct ibv_sge sge;
    sge.addr = reinterpret_cast<uint64_t>(ctx->buffer->addr);
    sge.length = kMempoolChunkSize;
    sge.lkey = ctx->buffer->lkey;

    wr.wr_id = reinterpret_cast<uint64_t>(ctx);
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    CHECK_EQ(ibv_post_srq_recv(srq_, &wr, &bad_wr), 0)
        << "ibv_post_srq_recv failed.";
  }

  void RegisterEndpointQP(Endpoint* endpoint) {
    std::lock_guard<std::mutex> lk(qp_map_mu_);
    qp_num_to_endpoint_[endpoint->cm_id->qp->qp_num] = endpoint;
  }

  void UnregisterEndpointQP(Endpoint* endpoint) {
    if (!endpoint || !endpoint->cm_id || !endpoint->cm_id->qp) {
      return;
    }
    std::lock_guard<std::mutex> lk(qp_map_mu_);
    qp_num_to_endpoint_.erase(endpoint->cm_id->qp->qp_num);
  }

  Endpoint* LookupEndpointByQPN(uint32_t qp_num) {
    std::lock_guard<std::mutex> lk(qp_map_mu_);
    auto it = qp_num_to_endpoint_.find(qp_num);
    CHECK(it != qp_num_to_endpoint_.end()) << "QP not registered: " << qp_num;
    return it->second;
  }

  void ReleaseWorkRequestContext(WRContext* context, Endpoint* endpoint) {
    switch (context->type) {
      case kRendezvousStartContext:
        endpoint->free_start_ctx.Push(context);
        break;
      case kRendezvousReplyContext:
        endpoint->free_reply_ctx.Push(context);
        break;
      case kReceiveContext:
        if (use_srq_) {
          PostSharedRecv(context);
        } else {
          endpoint->PostRecv(context);
        }
        break;
      default:
        CHECK(0);
    }
  }

  void PollCQ() {
    // Pre-allocated work completions array used for polling
    struct ibv_wc wc[kMaxConcurrentWorkRequest];
    while (!should_stop_.load()) {
      int ne = ibv_poll_cq(cq_, kMaxConcurrentWorkRequest, wc);
      CHECK_GE(ne, 0);
      for (int i = 0; i < ne; ++i) {
        CHECK(wc[i].status == IBV_WC_SUCCESS)
            << "Failed status \n"
            << ibv_wc_status_str(wc[i].status) << " " << wc[i].status << " "
            << static_cast<uint64_t>(wc[i].wr_id) << " " << wc[i].vendor_err
            << " postoffice ptr: " << (void*)postoffice_;

        switch (wc[i].opcode) {
          case IBV_WC_SEND: {
            WRContext* context = reinterpret_cast<WRContext*>(wc[i].wr_id);
            Endpoint* endpoint =
                reinterpret_cast<Endpoint*>(context->private_data);
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          case IBV_WC_RDMA_WRITE: {
            auto* msg_buf = reinterpret_cast<MessageBuffer*>(wc[i].wr_id);
            if (msg_buf && msg_buf->release_mrs_on_completion) {
              msg_buf->mrs.clear();
            }
          } break;
          case IBV_WC_RECV_RDMA_WITH_IMM: {
            WRContext* context = reinterpret_cast<WRContext*>(wc[i].wr_id);
            Endpoint* endpoint =
                use_srq_ ? LookupEndpointByQPN(wc[i].qp_num)
                         : reinterpret_cast<Endpoint*>(context->private_data);
            uint32_t addr_idx = wc[i].imm_data;
            BufferContext* buf_ctx = addr_pool_.GetAddress(addr_idx);
            recv_buffers_.Push(std::make_tuple(endpoint, buf_ctx));
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          case IBV_WC_RECV: {
            WRContext* context = reinterpret_cast<WRContext*>(wc[i].wr_id);
            Endpoint* endpoint =
                use_srq_ ? LookupEndpointByQPN(wc[i].qp_num)
                         : reinterpret_cast<Endpoint*>(context->private_data);
            CHECK(wc[i].wc_flags & IBV_WC_WITH_IMM);
            uint32_t imm = wc[i].imm_data;
            struct ibv_mr* mr = context->buffer;

            if (imm == kRendezvousStart) {
              RendezvousStart* req =
                  reinterpret_cast<RendezvousStart*>(mr->addr);
              auto trans = CHECK_NOTNULL(endpoint->GetTransport());
              trans->SendRendezvousReply(req, addr_pool_);

            } else if (imm == kRendezvousReply) {
              RendezvousReply* resp =
                  reinterpret_cast<RendezvousReply*>(mr->addr);
              uint64_t remote_addr = resp->addr;
              uint64_t origin_addr = resp->origin_addr;
              uint32_t rkey = resp->rkey;
              uint32_t idx = resp->idx;

              MessageBuffer* msg_buf =
                  reinterpret_cast<MessageBuffer*>(origin_addr);

              // Before RDMA write, store the remote info so that
              // subsequent write does not need repeated rendezvous
              StoreRemoteAndLocalInfo(msg_buf, remote_addr, rkey, idx);

              Message* msg = GetFirstMsg(msg_buf);

              auto addr_tuple = GetRemoteAndLocalInfo(
                  msg->meta.key, msg->meta.push, msg->meta.recver);

              PrintSendLog(*msg, msg_buf, addr_tuple);

              auto trans = CHECK_NOTNULL(endpoint->GetTransport());
              if (!IsValidPushpull(*msg)) {
                // control message
                msg_buf->release_mrs_on_completion = true;
                trans->RDMAWriteWithImm(msg_buf, remote_addr, rkey, idx);
              } else if (msg->meta.push && msg->meta.request) {
                // worker, push request
                msg_buf->release_mrs_on_completion = true;
                (void)KeysFromHeldMR(msg_buf, msg_buf->data[1].data(),
                                      msg_buf->data[1].size(), msg->meta.key,
                                      "push-request");
                trans->SendPushRequest(*msg, msg_buf, addr_tuple);
              } else if (msg->meta.push && !msg->meta.request) {
                // server, push response
                msg_buf->release_mrs_on_completion = true;
                trans->SendPushResponse(*msg, msg_buf, addr_tuple);
              } else if (!msg->meta.push && msg->meta.request) {
                // worker, pull request
                msg_buf->release_mrs_on_completion = false;
                auto keys = KeysFromHeldMR(
                    msg_buf, reinterpret_cast<char*>(msg->meta.addr),
                    msg->meta.val_len, msg->meta.key, "pull-request");
                msg->meta.option = keys.rkey;
                RepackMsgBufMeta(*msg, msg_buf);
                trans->SendPullRequest(*msg, msg_buf, addr_tuple);
              } else if (!msg->meta.push && !msg->meta.request) {
                // server, pull response
                msg_buf->release_mrs_on_completion = true;
                auto keys = KeysFromHeldMR(
                    msg_buf, msg_buf->data[1].data(), msg_buf->data[1].size(),
                    msg->meta.key, "pull-response");
                trans->SendPullResponse(*msg, msg_buf, addr_tuple, keys.lkey);
              }

              // release the msg_buf from msgbuf_cache_
              ReleaseFirstMsg(msg_buf);

            } else {
              CHECK(0);
            }
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          default:
            CHECK(0) << "Unexpected opcode: " << wc[i].opcode;
        }
      }
    }
  }

  void PollEvents() {
    int flags = fcntl(event_channel_->fd, F_GETFL);
    int rc = fcntl(event_channel_->fd, F_SETFL, flags | O_NONBLOCK);
    CHECK_GE(rc, 0);
    int error_flags = POLLERR | POLLHUP | POLLNVAL;

    while (!should_stop_.load()) {
      struct pollfd pfd = {
          .fd = event_channel_->fd, .events = POLLIN, .revents = 0};
      int ret = poll(&pfd, 1, 10);

      CHECK_GE(ret, 0) << strerror(errno);
      CHECK_EQ(pfd.revents & error_flags, 0);

      if (!(pfd.revents & POLLIN)) {
        continue;
      }

      struct rdma_cm_event* event;
      CHECK_EQ(rdma_get_cm_event(event_channel_, &event), 0);
      // TODO(clan): Reorder the list according to the event frequency
      switch (event->event) {
        case RDMA_CM_EVENT_CONNECT_REQUEST:
          OnConnectRequest(event);
          break;
        case RDMA_CM_EVENT_ADDR_RESOLVED:
          OnAddrResolved(event);
          break;
        case RDMA_CM_EVENT_ROUTE_RESOLVED:
          OnRouteResolved(event);
          break;
        case RDMA_CM_EVENT_ESTABLISHED:
          OnConnected(event);
          break;
        case RDMA_CM_EVENT_DISCONNECTED:
          OnDisconnected(event);
          break;
        case RDMA_CM_EVENT_REJECTED:
          OnRejected(event);
          break;
        default:
          CHECK(0) << "OnEvent: unknown event " << event->event << " ("
                   << rdma_event_str(event->event) << ")";
      }
      rdma_ack_cm_event(event);
    }
  }

  void OnRejected(struct rdma_cm_event* event) {
    struct rdma_cm_id* id = event->id;
    Endpoint* endpoint = reinterpret_cast<Endpoint*>(id->context);

    endpoints_mu_.lock();
    if (endpoint->isDataPlane) {
      auto it = data_endpoints_.find(endpoint->node_id);
      CHECK(it != data_endpoints_.end()) << "Connection not ready.";
    } else {
      auto it = endpoints_.find(endpoint->node_id);
      CHECK(it != endpoints_.end()) << "Connection not ready.";
    }

    endpoints_mu_.unlock();

    CHECK_EQ(endpoint->status, Endpoint::CONNECTING);
    CHECK_EQ(endpoint->cm_id, id);

    PS_VLOG(1) << my_node_.id << " to " << endpoint->node_id
               << " connection rejected, retrying...";
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->status = Endpoint::REJECTED;
    }
    endpoint->cv.notify_all();
  }

  void OnConnectRequest(struct rdma_cm_event* event) {
    struct rdma_cm_id* id = event->id;
    CHECK_NOTNULL(id);

    CHECK_LE(sizeof(RequestContext), event->param.conn.private_data_len)
        << "RequestContext size mismatch. Actual: "
        << (size_t)event->param.conn.private_data_len
        << ", Expected: " << sizeof(RequestContext);
    CHECK_NOTNULL(event->param.conn.private_data);

    const RequestContext* remote_ctx =
        reinterpret_cast<const RequestContext*>(event->param.conn.private_data);

    std::pair<std::unordered_set<std::unique_ptr<Endpoint>>::iterator, bool> r;
    {
      std::lock_guard<std::mutex> incoming_lk(incoming_mu_);
      r = incoming_.emplace(
          std::make_unique<Endpoint>(remote_ctx->isDataPlane));
    }
    Endpoint* endpoint = r.first->get();
    endpoint->SetNodeID(remote_ctx->node);
    endpoint->cm_id = id;
    id->context = endpoint;

    if (context_ == nullptr) {
      InitContext(id->verbs);
    }

    endpoint->Init(cq_, pd_, srq_);
    RegisterEndpointQP(endpoint);

    bool is_local_node =
        disable_ipc_
            ? false
            : (std::string(remote_ctx->hostname) == my_node_.hostname ? true
                                                                      : false);
    {
      std::lock_guard<std::mutex> lk(local_mu_);
      is_local_[remote_ctx->node] = is_local_node;
    }
    LOG(INFO) << my_node_.id << " OnConnect to " << remote_ctx->node
              << " with Transport=" << (is_local_node ? "IPC" : "RDMA")
              << ";DataPlane=" << (endpoint->isDataPlane ? "True" : "False");

    std::shared_ptr<Transport> t =
        is_local_node ? std::make_shared<IPCTransport>(
                            endpoint, mem_allocator_.get(), postoffice_)
                      : std::make_shared<RDMATransport>(
                            endpoint, mem_allocator_.get(), postoffice_);
    endpoint->SetTransport(t);

    RequestContext ctx;
    ctx.node = static_cast<uint32_t>(my_node_.id);
    ctx.port = static_cast<uint16_t>(my_node_.port);
    if (endpoint->isDataPlane) {
      ctx.isDataPlane = (uint8_t)1;
    } else {
      ctx.isDataPlane = (uint8_t)0;
    }
    snprintf(ctx.hostname, kMaxHostnameLength, "%s", my_node_.hostname.c_str());

    struct rdma_conn_param cm_params;
    memset(&cm_params, 0, sizeof(cm_params));
    cm_params.retry_count = 7;
    cm_params.rnr_retry_count = 7;
    cm_params.private_data = &ctx;
    cm_params.private_data_len = sizeof(RequestContext);

    CHECK_EQ(rdma_accept(id, &cm_params), 0)
        << "Accept RDMA connection failed: " << strerror(errno);
  }

  // Resolve a route after address is resolved
  void OnAddrResolved(struct rdma_cm_event* event) {
    struct rdma_cm_id* id = event->id;
    CHECK_EQ(rdma_resolve_route(id, kTimeoutms), 0)
        << "Resolve RDMA route failed";
  }

  // Make a connection after route is resolved
  void OnRouteResolved(struct rdma_cm_event* event) {
    struct rdma_cm_id* id = event->id;
    Endpoint* endpoint = reinterpret_cast<Endpoint*>(id->context);

    if (context_ == nullptr) {
      InitContext(id->verbs);
    }

    endpoint->Init(cq_, pd_, srq_);
    RegisterEndpointQP(endpoint);

    RequestContext ctx;
    ctx.node = static_cast<uint32_t>(my_node_.id);
    ctx.port = static_cast<uint16_t>(my_node_.port);
    if (endpoint->isDataPlane) {
      ctx.isDataPlane = (uint8_t)1;
    } else {
      ctx.isDataPlane = (uint8_t)0;
    }
    snprintf(ctx.hostname, kMaxHostnameLength, "%s", my_node_.hostname.c_str());

    struct rdma_conn_param cm_params;
    memset(&cm_params, 0, sizeof(cm_params));
    cm_params.retry_count = 7;
    cm_params.rnr_retry_count = 7;
    cm_params.private_data = &ctx;
    cm_params.private_data_len = sizeof(RequestContext);

    CHECK_EQ(rdma_connect(id, &cm_params), 0)
        << "RDMA connect failed" << strerror(errno);
  }

  void OnConnected(struct rdma_cm_event* event) {
    struct rdma_cm_id* id = event->id;
    CHECK(id) << "rdma_cm_id not found.";
    Endpoint* endpoint = reinterpret_cast<Endpoint*>(id->context);
    CHECK(endpoint) << "Endpoint not found.";

    if (cq_polling_thread_ == nullptr) {
      cq_polling_thread_.reset(new std::thread(&RDMAVan::PollCQ, this));
    }

    CHECK_EQ(endpoint->cm_id, id);
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->status = Endpoint::CONNECTED;
    }
    endpoint->cv.notify_all();
    if (endpoint->node_id != my_node_.id) {
      PS_VLOG(1) << my_node_.id << " OnConnected to " << endpoint->node_id;
    }
  }

  void OnDisconnected(struct rdma_cm_event* event) {
    struct rdma_cm_id* id = event->id;
    Endpoint* endpoint = reinterpret_cast<Endpoint*>(id->context);
    UnregisterEndpointQP(endpoint);
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->status = Endpoint::IDLE;
    }
    endpoint->cv.notify_all();
    LOG(INFO) << my_node_.id << " OnDisconnected from " << endpoint->node_id;
  }

  AddressPool<BufferContext> addr_pool_;
  std::unique_ptr<MemoryAllocator> mem_allocator_;

  std::unique_ptr<RDMATransport> rdma_trans_;
  std::unique_ptr<IPCTransport> ipc_trans_;

  struct rdma_cm_id* listener_ = nullptr;
  std::atomic<bool> should_stop_;

  std::mutex endpoints_mu_;
  std::unordered_map<int, std::unique_ptr<Endpoint>> endpoints_;
  std::unordered_map<int, std::unique_ptr<Endpoint>> data_endpoints_;
  std::mutex incoming_mu_;
  std::unordered_set<std::unique_ptr<Endpoint>> incoming_;

  struct rdma_event_channel* event_channel_ = nullptr;
  struct ibv_context* context_ = nullptr;

  // ibverbs protection domain
  struct ibv_pd* pd_ = nullptr;
  struct ibv_srq* srq_ = nullptr;
  WRContext* shared_rx_ctx_ = nullptr;
  bool use_srq_ = true;
  int srq_depth_ = 4096;
  // Completion event channel, to wait for work completions
  struct ibv_comp_channel* comp_event_channel_ = nullptr;
  // Completion queue, to poll on work completions
  struct ibv_cq* cq_ = nullptr;
  // cq thread
  std::unique_ptr<std::thread> cq_polling_thread_;
  // event thread
  std::unique_ptr<std::thread> cm_event_polling_thread_;
  // Recv buffer queue
  ThreadsafeQueue<std::tuple<Endpoint*, BufferContext*>> recv_buffers_;

  // local IPC related
  bool disable_ipc_ = false;
  std::mutex local_mu_;
  std::unordered_map<int, bool> is_local_;
  std::mutex qp_map_mu_;
  std::unordered_map<uint32_t, Endpoint*> qp_num_to_endpoint_;

  std::mutex addr_mu_;
  // <key, recver>, (<remote_addr, rkey, idx, local_addr>)
  std::unordered_map<uint64_t, RemoteAndLocalAddress> push_addr_;
  std::unordered_map<uint64_t, RemoteAndLocalAddress> pull_addr_;
  std::unordered_map<MessageBuffer*, Message> msgbuf_cache_;  // msg_buf, msg

  std::mutex map_mu_;
  std::condition_variable mr_cache_cv_;
  std::unordered_map<char*, struct ibv_mr*>
      mem_mr_;  // (memory address, ibv_mr)
  std::unordered_map<struct ibv_mr*, CachedMrInfo> mr_cache_info_;
  std::deque<struct ibv_mr*> mr_lru_;
  size_t mr_cache_budget_bytes_ = 192UL * 1024UL * 1024UL;
  size_t mr_host_cache_budget_bytes_ = 1024UL * 1024UL * 1024UL;
  size_t mr_cache_active_bytes_ = 0;
  size_t mr_device_cache_active_bytes_ = 0;
  size_t mr_host_cache_active_bytes_ = 0;
  uint64_t mr_lru_clock_ = 0;

  // logging
  bool enable_log_;
  bool enable_mr_debug_ = false;
  bool enable_reg_backoff_ = false;
  std::mutex log_mu_;

  int kMaxConcurrentWorkRequest = 4224;  // 128 + 2048 * 2

};  // class RDMAVan

};  // namespace ps

#endif  // DMLC_USE_RDMA
#endif  // PS_RDMA_VAN_H_
