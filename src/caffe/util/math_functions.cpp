//
// Created by yipeng on 2020/3/21.
//
#include <limits>
#include <random>
#include <iomanip>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//矩阵相乘
template <>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
													 const int M, const int N, const int K,
                           const float alpha, const float* A, const float* B,
                           const float beta, float* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K,
		          alpha, A, lda, B, ldb, beta, C, N);
}

//矩阵相乘
template <>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                            const int M, const int N, const int K,
                            const double alpha, const double* A, const double* B,
                            const double beta, double* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K,
	            alpha, A, lda, B, ldb, beta, C, N);
}

//矩阵和向量相乘
template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                           const float alpha, const float* A, const float* B,
                           const float beta, float* C) {
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N,
		          B, 1, beta, C, 1);
}

//矩阵和向量相乘
template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                            const double alpha, const double* A, const double* B,
                            const double beta, double* C) {
	cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N,
	            B, 1, beta, C, 1);
}

//y = ax + y
template <>
void caffe_axpy<float>(const int N, const float alpha,
                       const float* X, float* Y) {
	cblas_saxpy(N, alpha, X, 1, Y, 1);
}

//y = ax + y
template <>
void caffe_axpy<double>(const int N, const double alpha,
                        const double* X, double* Y) {
	cblas_daxpy(N, alpha, X, 1, Y, 1);
}

//y = ax + by
template <>
void caffe_cpu_axpby<float>(const int N, const float alpha,
                            const float* X, const float beta, float* Y) {
	cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

//y = ax + by
template <>
void caffe_cpu_axpby<double>(const int N, const double alpha,
                             const double* X, const double beta, double* Y) {
	cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

//拷贝
template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
	if (X != Y) {
		if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
			NO_GPU;
#endif
		} else {
			memcpy(Y, X, sizeof(Dtype) * N);
		}
	}
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

//设置为一个值
template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
	if (0 == alpha) {
		memset(Y, 0, sizeof(Dtype) * N);
		return ;
	}
	for (int i = 0; i < N; ++i) {
		Y[i] = alpha;
	}
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

//y = y + a
template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
	for (int i = 0; i < N; ++i) {
		Y[i] += alpha;
	}
}

//y = y + a
template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
	for (int i = 0; i < N; ++i) {
		Y[i] += alpha;
	}
}

//y = ay
template <>
void caffe_scal<float>(const int N, const float alpha, float* Y) {
	cblas_sscal(N, alpha, Y, 1);
}

//y = ay
template <>
void caffe_scal<double>(const int N, const double alpha, double* Y) {
	cblas_dscal(N, alpha, Y, 1);
}

//c = a + b
template <>
void caffe_add<float>(const int N, const float* A, const float* B, float* C) {
	vsAdd(N, A, B, C);
}

//c = a + b
template <>
void caffe_add<double>(const int N, const double* A, const double* B, double* C) {
	vdAdd(N, A, B, C);
}

//c = a - b
template <>
void caffe_sub<float>(const int N, const float* A, const float* B, float* C) {
	vsSub(N, A, B, C);
}

//c = a - b
template <>
void caffe_sub<double>(const int N, const double* A, const double* B, double* C) {
	vdSub(N, A, B, C);
}

//c = a * b
template <>
void caffe_mul<float>(const int N, const float* A, const float* B, float* C) {
	vsMul(N, A, B, C);
}

//c = a * b
template <>
void caffe_mul<double>(const int N, const double* A, const double* B, double* C) {
	vdMul(N, A, B, C);
}

//c = a / b
template <>
void caffe_div<float>(const int N, const float* A, const float* B, float* C) {
	vsDiv(N, A, B, C);
}

//c = a / b
template <>
void caffe_div<double>(const int N, const double* A, const double* B, double* C) {
	vdDiv(N, A, B, C);
}

//y = x * x
template <>
void caffe_square<float>(const int N, const float* X, float* Y) {
	vsSquare(N, X, Y);
}

//y = x * x
template <>
void caffe_square<double>(const int N, const double* X, double* Y) {
	vdSquare(N, X, Y);
}

//y = x开方
template <>
void caffe_sqrt<float>(const int N, const float* X, float* Y) {
	vsSqrt(N, X, Y);
}

//y = x开方
template <>
void caffe_sqrt<double>(const int N, const double* X, double* Y) {
	vdSqrt(N, X, Y);
}

//y = e的x次方
template <>
void caffe_exp<float>(const int N, const float* X, float* Y) {
	vsExp(N, X, Y);
}

//y = e的x次方
template <>
void caffe_exp<double>(const int N, const double* X, double* Y) {
	vdExp(N, X, Y);
}

//y = lnx log以10为底 x的对数
template <>
void caffe_log<float>(const int N, const float* X, float* Y) {
	vsLn(N, X, Y);
}

//y = lnx log以10为底 x的对数
template <>
void caffe_log<double>(const int N, const double* X, double* Y) {
	vdLn(N, X, Y);
}

//y = |x|
template <>
void caffe_abs<float>(const int N, const float* X, float* Y) {
	vsAbs(N, X, Y);
}

//y = |x|
template <>
void caffe_abs<double>(const int N, const double* X, double* Y) {
	vdAbs(N, X, Y);
}

//c = a的b次方
template <>
void caffe_pow<float>(const int N, const float* A, const float B, float* C) {
	vsPow(N, A, B, C);
}

//c = a的b次方
template <>
void caffe_pow<double>(const int N, const double* A, const double B, double* C) {
	vdPow(N, A, B, C);
}

//x步长取incx y步长取incy 得到向量的点积
template <>
float caffe_cpu_stride_dot<float>(const int N, const float* X, const int incx,
                                  const float* Y, const int incy) {
	return cblas_sdot(N, X, incx, Y, incy);
}

//x步长取incx y步长取incy 得到向量的点积
template <>
double caffe_cpu_stride_dot<double>(const int N, const double* X, const int incx,
	                                  const double* Y, const int incy) {
	return cblas_ddot(N, X, incx, Y, incy);
}

//x步长取1 y步长取1 得到向量的点积
template <typename Dtype>
Dtype caffe_cpu_dot(const int N, const Dtype* X, const Dtype* Y) {
	return caffe_cpu_stride_dot(N, X, 1, Y, 1);
}

template float caffe_cpu_dot<float>(const int N, const float* X, const float* Y);
template double caffe_cpu_dot<double>(const int N, const double* X, const double* Y);

//求和
template <>
float caffe_cpu_asum<float>(const int N, const float* Y) {
	return cblas_sasum(N, Y, 1);
}

//求和
template <>
double caffe_cpu_asum<double>(const int N, const double* Y) {
	return cblas_dasum(N, Y, 1);
}

//y = ax
template <>
void caffe_cpu_scale<float>(const int N, const float alpha, const float* X, float* Y) {
	cblas_scopy(N, X, 1, Y, 1);
	cblas_sscal(N, alpha, Y, 1);
}

//y = ax
template <>
void caffe_cpu_scale<double>(const int N, const double alpha, const double* X, double* Y) {
	cblas_dcopy(N, X, 1, Y, 1);
	cblas_dscal(N, alpha, Y, 1);
}

//均匀分布 a是下界 b是上界
template <typename Dtype>
void caffe_rng_uniform(const int N, const Dtype a, const Dtype b, Dtype* c) {
	CHECK_GE(N, 0);
	CHECK_LE(a, b);
	CHECK(c);
//	std::random_device rand_device;
//	std::default_random_engine random_engine(rand_device());
	std::uniform_real_distribution<Dtype> random_generator(a, b);
	for (int i = 0; i < N; ++i) {
		c[i] = random_generator(caffe_rng());
	}
}

template void caffe_rng_uniform<float>(const int N, const float a, const float b, float* c);
template void caffe_rng_uniform<double>(const int N, const double a, const double b, double* c);

//均匀分布 得到整数值 a是下界 b是上界
template <typename Dtype>
void caffe_rng_uniform_int(const int N, const Dtype a, const Dtype b, int* c) {
	CHECK_GE(N, 0);
	CHECK_LE(a, b);
	CHECK(c);
//	std::random_device rand_device;
//	std::default_random_engine random_engine(rand_device());
	std::uniform_int_distribution<int> random_generator(a, b);
	for (int i = 0; i < N; ++i) {
		c[i] = random_generator(caffe_rng());
	}
}

template void caffe_rng_uniform_int<float>(const int N, const float a, const float b, int* c);
template void caffe_rng_uniform_int<double>(const int N, const double a, const double b, int* c);

//高斯分布 也是正态分布 均值 标准差
template <typename Dtype>
void caffe_rng_gaussian(const int N, const Dtype mean, const Dtype stddev, Dtype* c) {
	CHECK_GE(N, 0);
	CHECK_GT(stddev, 0);
	CHECK(c);
//	std::random_device rand_device;
//	std::default_random_engine random_engine(rand_device());
	std::normal_distribution<Dtype> random_generator(mean, stddev);
	for (int i = 0; i < N; ++i) {
		c[i] = random_generator(caffe_rng());
	}
}

template void caffe_rng_gaussian<float>(const int N, const float mean, const float stddev, float* c);
template void caffe_rng_gaussian<double>(const int N, const double mean, const double stddev, double* c);

//伯努利分布 是实验次数为1的二项分布 值只有0和1  p代表为1的概率 1-p是为0的概率
template <typename Dtype>
void caffe_rng_bernoulli(const int N, const Dtype p, int* c) {
	CHECK_GE(N, 0);
	CHECK_GE(p, 0);
	CHECK_LE(p, 1);
	CHECK(c);
//	std::random_device rand_device;
//	std::default_random_engine random_engine(rand_device());
	std::bernoulli_distribution random_generator(p);
	for (int i = 0; i < N; ++i) {
		c[i] = random_generator(caffe_rng());
	}
}

template void caffe_rng_bernoulli<float>(const int N, const float p, int* c);
template void caffe_rng_bernoulli<double>(const int N, const double p, int* c);

template <typename Dtype>
void caffe_show(const int N, const Dtype* c) {
	for (int i = 0; i < N; ++i) {
		int space_number = 0;
		if (c[i] >= 0) {
			if ((c[i] / 10.0) < 1.0) {
				space_number = 3;
			} else {
				space_number = 2;
			}
		} else {
			if ((c[i] / 10.0) < -1.0) {
				space_number = 1;
			} else {
				space_number = 2;
			}
		}

		std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
							<< std::setprecision(6) << c[i]
							<< std::string(space_number, ' ');
		if (0 == (i + 1) % 10) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

template void caffe_show<int>(const int N, const int* c);
template void caffe_show<float>(const int N, const float* c);
template void caffe_show<double>(const int N, const double* c);
}     //namespace caffe

//int main() {
//	float a[10] = {0};
//	int b[10] = {0};
//	float c[10] = {0};
//	int d[10] = {0};
//
//	float q = 0,w=10;
//	float p = 0.5;
//	caffe::caffe_rng_uniform(10, q,w,a);
//	caffe::caffe_rng_uniform_int(10, q,w,b);
//	caffe::caffe_rng_gaussian(10, q,w,c);
//	caffe::caffe_rng_bernoulli(10,p,d);
//
//	caffe::caffe_show(10,a);
//	caffe::caffe_show(10,b);
//	caffe::caffe_show(10,c);
//	caffe::caffe_show(10,d);

//	float A[6] = {1, 2, 3, 4, 5, 6};
//	float B[2] = {7, 8};
//	float C[3] = {0.5};
//	float a=1.0,b=0;
//	caffe::caffe_cpu_gemv(CblasNoTrans,3,2,a,A,B,b,C);
//
//	caffe::caffe_show(3,C);
//}