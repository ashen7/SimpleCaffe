//
// Created by yipeng on 2020/3/21.
//
#ifndef SIMPLE_CAFFE_UTILITY_MATH_FUNCTIONS_HPP_
#define SIMPLE_CAFFE_UTILITY_MATH_FUNCTIONS_HPP_

#include <glog/logging.h>

#include <cstdint>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/host_alternate.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {
//cpu 用的openblas来实现矩阵 向量运算
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
										const int M, const int N, const int K,
										const Dtype alpha, const Dtype* A, const Dtype* B,
										const Dtype beta, Dtype* C);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype* A, const Dtype* B,
                    const Dtype beta, Dtype* C);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha,
								const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha,
                     const Dtype* X, const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y);

inline void caffe_memset(const int N, const int alpha, void* Y) {
	memset(Y, alpha, N);
}

template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype* Y);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype* Y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_div(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_square(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_sqrt(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_exp(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_log(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_abs(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_pow(const int N, const Dtype* A, const Dtype B, Dtype* C);

template <typename Dtype>
Dtype caffe_cpu_stride_dot(const int N, const Dtype* X, const int incx,
	                         const Dtype* Y, const int incy);

template <typename Dtype>
Dtype caffe_cpu_dot(const int N, const Dtype* X, const Dtype* Y);

template <typename Dtype>
Dtype caffe_cpu_asum(const int N, const Dtype* Y);

template <typename Dtype>
void caffe_cpu_scale(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

//内联函数 实现可以写在头文件 普通函数会报重定义
inline Caffe::rng_t& caffe_rng() {
	return (*static_cast<Caffe::rng_t*>(Caffe::rng_stream().generator()));
}

inline unsigned int caffe_rng_rand() {
	return caffe_rng()();
}

template <typename Dtype>
void caffe_rng_uniform(const int N, const Dtype a, const Dtype b, Dtype* c);

template <typename Dtype>
void caffe_rng_uniform_int(const int N, const Dtype a, const Dtype b, int* c);

template <typename Dtype>
void caffe_rng_gaussian(const int N, const Dtype mean, const Dtype stddev, Dtype* c);

template <typename Dtype>
void caffe_rng_bernoulli(const int N, const Dtype p, int* c);

template <typename Dtype>
void caffe_show(const int N, const Dtype* c);

template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
	return (Dtype(0) < val) - (val < Dtype(0));
}

#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
  CHECK_GT(n, 0); CHECK(x); CHECK(y); \
  for (int i = 0; i < n; ++i) { \
    operation; \
  } \
}

// x为positives输出1, 为zero输出0, 为negative输出-1
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]))
// x为负数输出1 其他输出0
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, y[i] = static_cast<bool>((std::signbit)(x[i])))
DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))


#ifndef CPU_ONLY  // GPU + CPU Mode
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
										const int M, const int N, const int K,
                    const Dtype alpha, const Dtype* A, const Dtype* B,
                    const Dtype beta, Dtype* C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype* A, const Dtype* B,
                    const Dtype beta, Dtype* C);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha,
										const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha,
										 const Dtype* X, const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y);

void caffe_gpu_memcpy(const int N, const void* X, void* Y);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* Y) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(Y, alpha, N));
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype* Y);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype* Y);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype* X, cudaStream_t str);

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* A, const Dtype* B, Dtype* C);

template <typename Dtype>
void caffe_gpu_square(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_sqrt(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_exp(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_log(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_abs(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_pow(const int N, const Dtype* A, const Dtype B, Dtype* C);

template <typename Dtype>
void caffe_gpu_dot(const int N, const Dtype* X, const Dtype* Y, Dtype* output);

template <typename Dtype>
void caffe_gpu_asum(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_scale(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

void caffe_gpu_rng_uniform(const int N, unsigned int* c);

template <typename Dtype>
void caffe_gpu_rng_uniform(const int N, const Dtype a, const Dtype b, Dtype* c);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int N, const Dtype mean, const Dtype stddev, Dtype* c);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int N, const Dtype p, int* c);

template<typename Dtype>
void caffe_gpu_sign(const int N, const Dtype* X, Dtype* Y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_gpu_fabs(const int N, const Dtype* X, Dtype* Y);

#define DEFINE_AND_INSTANCE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif //!CPU_ONLY
}      //namespace caffe

#endif //SIMPLE_CAFFE_UTILITY_MATH_FUNCTIONS_HPP_
