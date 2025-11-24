#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// ----------------------------
// 基本别名 & 虚拟 THCState
// ----------------------------
struct THCState {};                       // 占位，不再需要真实状态
using THCudaTensor        = at::Tensor;   // float
using THCudaDoubleTensor  = at::Tensor;   // double
using THCudaLongTensor    = at::Tensor;   // int64
using THCudaIntTensor     = at::Tensor;   // int32
using THCudaByteTensor    = at::Tensor;   // uint8

// ----------------------------
// 常用检查/错误宏（与旧版风格相近）
// ----------------------------
#define THCudaCheck(err) TORCH_CHECK((err) == cudaSuccess, "CUDA error: ", cudaGetErrorString(err))
#define THCState_getCurrentStream(...) at::cuda::getCurrentCUDAStream()
// ----------------------------
// 设备与流（stream）相关
// ----------------------------
inline int THCudaGetCurrentDevice(THCState*) {
  return c10::cuda::current_device();
}

inline void THCudaCheckSetDevice(THCState*, int device) {
  c10::cuda::CUDAGuard g(device); // 切换当前设备
}

// inline cudaStream_t THCState_getCurrentStream(THCState*) {
//   return at::cuda::getCurrentCUDAStream().stream();
// }

inline cudaStream_t THCState_getCurrentStreamOnDevice(THCState*, int device) {
  // 你代码里用到的这个 API -> 映射到 ATen
  return at::cuda::getCurrentCUDAStream(device).stream();
}

// ----------------------------
// Tensor 基础属性（兼容常见旧 API）
// 旧 API 多以指针传入，这里也都用指针形参
// ----------------------------
inline ptrdiff_t THCudaTensor_nElement(THCState*, THCudaTensor* t) {
  return t->numel();
}
inline ptrdiff_t THCudaDoubleTensor_nElement(THCState*, THCudaDoubleTensor* t) {
  return t->numel();
}
inline ptrdiff_t THCudaLongTensor_nElement(THCState*, THCudaLongTensor* t) {
  return t->numel();
}
inline ptrdiff_t THCudaIntTensor_nElement(THCState*, THCudaIntTensor* t) {
  return t->numel();
}
inline ptrdiff_t THCudaByteTensor_nElement(THCState*, THCudaByteTensor* t) {
  return t->numel();
}

inline bool THCudaTensor_isContiguous(THCState*, THCudaTensor* t) {
  return t->is_contiguous();
}

// 尺寸/步长（按维度）
inline int64_t THCudaTensor_size(THCState*, THCudaTensor* t, int dim) {
  return t->size(dim);
}
inline int64_t THCudaTensor_stride(THCState*, THCudaTensor* t, int dim) {
  return t->stride(dim);
}

// ----------------------------
// data_ptr 映射（各 dtype）
// ----------------------------
inline float*    THCudaTensor_data(THCState*, THCudaTensor* t)               { return t->data_ptr<float>(); }
inline double*   THCudaDoubleTensor_data(THCState*, THCudaDoubleTensor* t)   { return t->data_ptr<double>(); }
inline int64_t*  THCudaLongTensor_data(THCState*, THCudaLongTensor* t)       { return t->data_ptr<int64_t>(); }
inline int32_t*  THCudaIntTensor_data(THCState*, THCudaIntTensor* t)         { return t->data_ptr<int32_t>(); }
inline uint8_t*  THCudaByteTensor_data(THCState*, THCudaByteTensor* t)       { return t->data_ptr<uint8_t>(); }

// ----------------------------
// 设备内存分配/释放（旧 THC 风格）
// ----------------------------
inline void THCudaMalloc(THCState*, void** ptr, size_t size) {
  THCudaCheck(cudaMalloc(ptr, size));
}
inline void THCudaFree(THCState*, void* ptr) {
  if (ptr) THCudaCheck(cudaFree(ptr));
}

// ----------------------------
// 同步/事件（如需要可继续扩充）
// ----------------------------
inline void THCudaSynchronize(THCState*) {
  THCudaCheck(cudaDeviceSynchronize());
}

// ----------------------------
// 若你代码里用到 “默认流/指定流” 的切换，这里给个便捷封装
// ----------------------------
inline cudaStream_t THCStream_defaultStream(THCState*, int device) {
  return at::cuda::getDefaultCUDAStream(device).stream();
}
