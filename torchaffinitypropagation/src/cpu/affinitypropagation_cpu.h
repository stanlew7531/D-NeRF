#ifndef _AFFINITYPROPAGATION_CPU
#define _AFFINITYPROPAGATION_CPU

#include <torch/extension.h>

std::vector<at::Tensor> affinitypropagation_cpu(at::Tensor similarities, int64_t iters = 1);

#endif