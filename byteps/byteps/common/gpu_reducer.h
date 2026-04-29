// Copyright 2021 Bytedance Inc. or its affiliates. All Rights Reserved.

#ifndef BYTEPS_GPU_REDUCER_H
#define BYTEPS_GPU_REDUCER_H

#if HAVE_CUDA == 1
#include <cuda_runtime.h>
#endif

#include "common.h"
#include "logging.h"

namespace byteps {
namespace common {

class GpuReducer {
 public:
  GpuReducer();
  ~GpuReducer();

  int copy(void* dst, bool to_gpu, const void* src, bool from_gpu,
           size_t len, bool async);
  int copy_h2d(void* dst, const void* src, size_t len, bool async);
  int copy_d2h(void* dst, const void* src, size_t len, bool async);
  int copy_d2d(void* dst, const void* src, size_t len, bool async);

  void sync_h2d();
  void sync_d2h();
  void sync_d2d();

 private:
  void InitStream();

#if HAVE_CUDA == 1
  cudaStream_t* _h2d_stream = nullptr;
  cudaStream_t* _d2h_stream = nullptr;
  cudaStream_t* _d2d_stream = nullptr;
#endif
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_GPU_REDUCER_H
