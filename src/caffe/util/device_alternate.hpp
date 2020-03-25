//
// Created by yipeng on 2020/3/21.
//
#ifndef SIMPLE_CAFFE_DEVICE_ALTERNATE_HPP_
#define SIMPLE_CAFFE_DEVICE_ALTERNATE_HPP_

#ifdef CPU_ONLY   //CPU Mode
#include <vector>

//cpu模式调用GPU直接FATAL
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-Only Caffe Mode: check mode"

#else //GPU + CPU Mode
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#ifdef USE_CUDNN
#include "cudnn.hpp"
#endif

//CUDA的函数调用检查宏 这里使用do while块 为了方便调用时最后面加一个;
#define CUDA_CHECK(condition) \
	do { \
		cudaError_t error = condition; \
		CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
	} while (0)

#define CUBLAS_CHECK(condition) \
	do { \
		cublasStatus_t status = condition; \
		CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " << caffe::cublasGetErrorString(status); \
	} while (0)

#define CURAND_CHECK(condition) \
	do { \
		curandStatus_t status = condition; \
		CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " << caffe::curandGetErrorString(status); \
	} while (0)

//cuda核函数的循环
#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	     i < (n); \
	     i += blockDim.x * gridDim.x)

//检查cuda核函数调用是否成功
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

//CUDA每块线程数 512
const int CAFFE_CUDA_NUM_THREADS = 512;
//根据总量 得到CUDA线程块数
inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

//得到cublas curand的 status函数
const char* cublasGetErrorString(cublasStatus_t status);
const char* curandGetErrorString(curandStatus_t status);

}      //namespace caffe

#endif //!CPU_ONLY
#endif //SIMPLE_CAFFE_DEVICE_ALTERNATE_HPP_
