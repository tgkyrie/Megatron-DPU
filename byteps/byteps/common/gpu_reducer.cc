// Copyright 2021 Bytedance Inc. or its affiliates. All Rights Reserved.

#include "gpu_reducer.h"

namespace byteps {
namespace common {

GpuReducer::GpuReducer() { InitStream(); }

GpuReducer::~GpuReducer() {
#if HAVE_CUDA == 1
  if (_h2d_stream) {
    CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
    CUDA_CALL(cudaStreamDestroy(*_h2d_stream));
    free(_h2d_stream);
  }
  if (_d2h_stream) {
    CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
    CUDA_CALL(cudaStreamDestroy(*_d2h_stream));
    free(_d2h_stream);
  }
  if (_d2d_stream) {
    CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
    CUDA_CALL(cudaStreamDestroy(*_d2d_stream));
    free(_d2d_stream);
  }
#endif
}

void GpuReducer::InitStream() {
#if HAVE_CUDA == 1
  _h2d_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  _d2h_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  _d2d_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  CUDA_CALL(cudaStreamCreateWithFlags(_h2d_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamCreateWithFlags(_d2h_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamCreateWithFlags(_d2d_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
  CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
  CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
#endif
}

int GpuReducer::copy(void* dst, bool to_gpu, const void* src, bool from_gpu,
                     size_t len, bool async) {
  BPS_CHECK(to_gpu || from_gpu) << to_gpu << " " << from_gpu;
  if (to_gpu && from_gpu) {
    return copy_d2d(dst, src, len, async);
  }
  if (to_gpu) {
    return copy_h2d(dst, src, len, async);
  }
  return copy_d2h(dst, src, len, async);
}

int GpuReducer::copy_d2d(void* dst, const void* src, size_t len, bool async) {
#if HAVE_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice,
                            *_d2d_stream));
  if (!async) CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
  return 0;
#else
  BPS_CHECK(0) << "GpuReducer requires HAVE_CUDA=1";
  return -1;
#endif
}

int GpuReducer::copy_h2d(void* dst, const void* src, size_t len, bool async) {
#if HAVE_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice,
                            *_h2d_stream));
  if (!async) CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
  return 0;
#else
  BPS_CHECK(0) << "GpuReducer requires HAVE_CUDA=1";
  return -1;
#endif
}

int GpuReducer::copy_d2h(void* dst, const void* src, size_t len, bool async) {
#if HAVE_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost,
                            *_d2h_stream));
  if (!async) CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
  return 0;
#else
  BPS_CHECK(0) << "GpuReducer requires HAVE_CUDA=1";
  return -1;
#endif
}

void GpuReducer::sync_h2d() {
#if HAVE_CUDA == 1
  CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
#endif
}

void GpuReducer::sync_d2h() {
#if HAVE_CUDA == 1
  CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
#endif
}

void GpuReducer::sync_d2d() {
#if HAVE_CUDA == 1
  CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
#endif
}

}  // namespace common
}  // namespace byteps
