#pragma once
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

using THByteTensor   = at::Tensor;
using THIntTensor    = at::Tensor;
using THLongTensor   = at::Tensor;
using THFloatTensor  = at::Tensor;
using THDoubleTensor = at::Tensor;