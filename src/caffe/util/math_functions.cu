//
// Created by yipeng on 2020/3/21.
//
#include <cmath>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
//矩阵相乘
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                           const int M, const int N, const int K,
                           const float alpha, const float* A, const float* B,
                           const float beta, float* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K,
	             &alpha, B, ldb, A, lda, &beta, C, N));
}

//矩阵相乘
template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                            const int M, const int N, const int K,
                            const double alpha, const double* A, const double* B,
                            const double beta, double* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K,
		           &alpha, B, ldb, A, lda, &beta, C, N));
}

//矩阵和向量相乘
template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                           const float alpha, const float* A, const float* B,
                           const float beta, float* C) {
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
		           A, N, B, 1, &beta, C, 1));
}

//矩阵和向量相乘
template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                            const double alpha, const double* A, const double* B,
                            const double beta, double* C) {
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
	             A, N, B, 1, &beta, C, 1));
}

//y = ax + y
template <>
void caffe_gpu_axpy<float>(const int N, const float alpha,
                           const float* X, float* Y) {
	CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

//y = ax + y
template <>
void caffe_gpu_axpy<double>(const int N, const double alpha,
                            const double* X, double* Y) {
	CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const int N, const void* X, void* Y) {
	if (X != Y) {
		CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
	}
}

//设置值的核函数
template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		y[thread_id] = alpha;
	}
}

//设置为一个值
template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
	if (0 == alpha) {
		CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));
		return ;
	}
	set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

//add一个值的核函数
template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		y[thread_id] += alpha;
	}
}

//y = y + a
template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
	add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
}

//y = y + a
template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
	add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
}

//y = ay
template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* Y) {
	CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, Y, 1));
}

//y = ay
template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* Y) {
	CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, Y, 1));
}

//y = ax + by
template <>
void caffe_gpu_axpby<float>(const int N, const float alpha,
                            const float* X, const float beta, float* Y) {
	caffe_gpu_scal<float>(N, beta, Y);
	caffe_gpu_axpy<float>(N, alpha, X, Y);
}

//y = ax + by
template <>
void caffe_gpu_axpby<double>(const int N, const double alpha,
                             const double* X, const double beta, double* Y) {
	caffe_gpu_scal<double>(N, beta, Y);
	caffe_gpu_axpy<double>(N, alpha, X, Y);
}

//add的核函数
template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
													 const Dtype*b, Dtype* c) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		c[thread_id] = a[thread_id] + b[thread_id];
	}
}

//c = a + b
template <>
void caffe_gpu_add<float>(const int N, const float* A, const float* B, float* C) {
	add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//c = a + b
template <>
void caffe_gpu_add<double>(const int N, const double* A, const double* B, double* C) {
	add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//sub的核函数
template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
                           const Dtype*b, Dtype* c) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		c[thread_id] = a[thread_id] - b[thread_id];
	}
}

//c = a - b
template <>
void caffe_gpu_sub<float>(const int N, const float* A, const float* B, float* C) {
	sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//c = a - b
template <>
void caffe_gpu_sub<double>(const int N, const double* A, const double* B, double* C) {
	sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//mul的核函数
template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
                           const Dtype*b, Dtype* c) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		c[thread_id] = a[thread_id] * b[thread_id];
	}
}

//c = a * b
template <>
void caffe_gpu_mul<float>(const int N, const float* A, const float* B, float* C) {
	mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//c = a * b
template <>
void caffe_gpu_mul<double>(const int N, const double* A, const double* B, double* C) {
	mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//div的核函数
template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
                           const Dtype*b, Dtype* c) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		c[thread_id] = a[thread_id] / b[thread_id];
	}
}

//c = a / b
template <>
void caffe_gpu_div<float>(const int N, const float* A, const float* B, float* C) {
	div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//c = a / b
template <>
void caffe_gpu_div<double>(const int N, const double* A, const double* B, double* C) {
	div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//square的核函数
template <typename Dtype>
__global__ void square_kernel(const int n, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		y[thread_id] = x[thread_id] * x[thread_id];
	}
}

//y = x * x
template <>
void caffe_gpu_square<float>(const int N, const float* X, float* Y) {
	square_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//y = x * x
template <>
void caffe_gpu_square<double>(const int N, const double* X, double* Y) {
	square_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//sqrt的核函数
template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		y[thread_id] = sqrt(x[thread_id]);
	}
}

//y = x开方
template <>
void caffe_gpu_sqrt<float>(const int N, const float* X, float* Y) {
	sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//y = x开方
template <>
void caffe_gpu_sqrt<double>(const int N, const double* X, double* Y) {
	sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//exp的核函数
template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		y[thread_id] = exp(x[thread_id]);
	}
}

//y = e的x次方
template <>
void caffe_gpu_exp<float>(const int N, const float* X, float* Y) {
	exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//y = e的x次方
template <>
void caffe_gpu_exp<double>(const int N, const double* X, double* Y) {
	exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//log的核函数
template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		y[thread_id] = log(x[thread_id]);
	}
}

//y = lnx log以10为底 x的对数
template <>
void caffe_gpu_log<float>(const int N, const float* X, float* Y) {
	log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//y = lnx log以10为底 x的对数
template <>
void caffe_gpu_log<double>(const int N, const double* X, double* Y) {
	log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//abs的核函数
template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		y[thread_id] = abs(x[thread_id]);
	}
}

//y = |x|
template <>
void caffe_gpu_abs<float>(const int N, const float* X, float* Y) {
	abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//y = |x|
template <>
void caffe_gpu_abs<double>(const int N, const double* X, double* Y) {
	abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, Y);
}

//pow的核函数
template <typename Dtype>
__global__ void pow_kernel(const int n, const Dtype* a, const Dtype b, Dtype* c) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		c[thread_id] = pow(a[thread_id], b);
	}
}

//c = a的b次方
template <>
void caffe_gpu_pow<float>(const int N, const float* A, const float B, float* C) {
	pow_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//c = a的b次方
template <>
void caffe_gpu_pow<double>(const int N, const double* A, const double B, double* C) {
	pow_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, A, B, C);
}

//x步长取1 y步长取1 得到向量的点积
template <>
void caffe_gpu_dot<float>(const int N, const float* X, const float* Y, float* output) {
	CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), N, X, 1, Y, 1, output));
}

template <>
void caffe_gpu_dot<double>(const int N, const double* X, const double* Y, double* output) {
	CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), N, X, 1, Y, 1, output));
}

//求和
template <>
void caffe_gpu_asum<float>(const int N, const float* X, float* Y) {
	CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), N, X, 1, Y));
}

//求和
template <>
void caffe_gpu_asum<double>(const int N, const double* X, double* Y) {
	CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), N, X, 1, Y));
}

//y = ax
template <>
void caffe_gpu_scale<float>(const int N, const float alpha, const float* X, float* Y) {
	CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), N, X, 1, Y, 1));
	CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, Y, 1));
}

//y = ax
template <>
void caffe_gpu_scale<double>(const int N, const double alpha, const double* X, double* Y) {
	CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), N, X, 1, Y, 1));
	CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, Y, 1));
}

DEFINE_AND_INSTANCE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
																	 - (x[index] < Dtype(0)));
DEFINE_AND_INSTANCE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int N, unsigned int* c) {
	CURAND_CHECK(curandGenerate(Caffe::curand_generator(), c, N));
}

//均匀分布 a是下界 b是上界
template <>
void caffe_gpu_rng_uniform<float>(const int N, const float a, const float b, float* c) {
	//生成0 - 1的值 然后每个值乘以b-a 范围就是0 b-a 在加a 就是a b
	CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), c, N));
	const float range = b - a;
	if (range != static_cast<float>(1)) {
		caffe_gpu_scal(N, range, c);
	}
	if (a != static_cast<float>(0)) {
		caffe_gpu_add_scalar(N, a, c);
	}
}

//均匀分布 a是下界 b是上界
template <>
void caffe_gpu_rng_uniform<double>(const int N, const double a, const double b, double* c) {
	//生成0 - 1的值 然后每个值乘以b-a 范围就是0 b-a 在加a 就是a b
	CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), c, N));
	const double range = b - a;
	if (range != static_cast<double>(1)) {
		caffe_gpu_scal(N, range, c);
	}
	if (a != static_cast<double>(0)) {
		caffe_gpu_add_scalar(N, a, c);
	}
}

//高斯分布 也是正态分布 均值 标准差
template <>
void caffe_gpu_rng_gaussian<float>(const int N, const float mean, const float stddev, float* c) {
	CURAND_CHECK(curandGenerateNormal(Caffe::curand_generator(), c, N, mean, stddev));
}
template <>
void caffe_gpu_rng_gaussian<double>(const int N, const double mean, const double stddev, double* c) {
	CURAND_CHECK(curandGenerateNormalDouble(Caffe::curand_generator(), c, N, mean, stddev));
}

}       //namespace caffe

//int main() {
////	caffe::Caffe::DeviceQuery();
//	using std::cout;
//	using std::endl;
//	unsigned int* d_A;
//	float* d_B;
//	float* d_C;
//	unsigned int A[10] = {0};
//	float B[10] = {0};
//	float C[10] = {0};
//
//	float alpha = 1.0;
//	float beta = 0.0;
//	cudaMalloc((void**)&d_A, sizeof(float)*10);
//	cudaMalloc((void**)&d_B, sizeof(float)*10);
//	cudaMalloc((void**)&d_C, sizeof(float)*10);
//
//	float q = 10;
//	float w = 20;
//	float e = 30;
//	caffe::caffe_gpu_rng_uniform(10,d_A);
//	caffe::caffe_gpu_rng_uniform(10, beta, q, d_B);
//	caffe::caffe_gpu_rng_gaussian(10, beta, q, d_C);
//
//	cudaMemcpy(A, d_A, sizeof(float)*10, cudaMemcpyDeviceToHost);
//	cudaMemcpy(B, d_B, sizeof(float)*10, cudaMemcpyDeviceToHost);
//	cudaMemcpy(C, d_C, sizeof(float)*10, cudaMemcpyDeviceToHost);
//	for (int i = 0; i < 10; i++){
//		std::cout <<A[i]<< "\t";
//	}
//	std::cout << std::endl;
//	for (int i = 0; i < 10; i++){
//		std::cout <<B[i]<< "\t";
//	}
//	std::cout << std::endl;
//	for (int i = 0; i < 10; i++){
//		std::cout <<C[i]<< "\t";
//	}
//	std::cout << std::endl;
//
//	cudaFree(d_A);
//	cudaFree(d_B);
//	cudaFree(d_C);
//
//	return 0;
//}