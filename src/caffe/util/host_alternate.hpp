//
// Created by yipeng on 2020/3/21.
//
#ifndef SIMPLE_CAFFE_UTILITY_HOST_ALTERNATE_HPP_
#define SIMPLE_CAFFE_UTILITY_HOST_ALTERNATE_HPP_

#include <cmath>

#include <glog/logging.h>
#include <cblas.h>

//一元函数
#define DEFINE_UNARY_FUNC(name, operation) \
	template <typename Dtype> \
	void v##name(const int n, const Dtype* x, Dtype* y) { \
		CHECK_GT(n, 0); CHECK(x); CHECK(y); \
		for (int i = 0; i < n; ++i) { \
			operation; \
		} \
	} \
	inline void vs##name(const int n, const float* x, float* y) { \
		v##name<float>(n, x, y); \
	} \
	inline void vd##name(const int n, const double* x, double* y) { \
		v##name<double>(n, x, y); \
	}

DEFINE_UNARY_FUNC(Square, y[i] = x[i] * x[i])
DEFINE_UNARY_FUNC(Sqrt,   y[i] = sqrt(x[i]))
DEFINE_UNARY_FUNC(Exp,    y[i] = exp(x[i]))
DEFINE_UNARY_FUNC(Ln,     y[i] = log(x[i]))
DEFINE_UNARY_FUNC(Abs,    y[i] = fabs(x[i]))

//一元函数带一个参数
#define DEFINE_UNARY_FUNC_WITH_PARAM(name, operation) \
	template <typename Dtype> \
	void v##name(const int n, const Dtype* x, const Dtype a, Dtype* y) { \
		CHECK_GT(n, 0); CHECK(x); CHECK(y); \
		for (int i = 0; i < n; ++i) { \
			operation; \
		} \
	} \
	inline void vs##name(const int n, const float* x, const float a, float* y) { \
		v##name<float>(n, x, a, y); \
	} \
	inline void vd##name(const int n, const double* x, const double a, double* y) { \
		v##name<double>(n, x, a, y); \
	}

DEFINE_UNARY_FUNC_WITH_PARAM(Pow, y[i] = pow(x[i], a))

//二元函数
#define DEFINE_BINARY_FUNC(name, operation) \
	template <typename Dtype> \
	void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* c) { \
		CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(c); \
		for (int i = 0; i < n; ++i) { \
			operation; \
		} \
	} \
	inline void vs##name(const int n, const float* a, const float* b, float* c) { \
		v##name<float>(n, a, b, c); \
	} \
	inline void vd##name(const int n, const double* a, const double* b, double* c) { \
		v##name<double>(n, a, b, c); \
	}

DEFINE_BINARY_FUNC(Add, c[i] = a[i] + b[i])
DEFINE_BINARY_FUNC(Sub, c[i] = a[i] - b[i])
DEFINE_BINARY_FUNC(Mul, c[i] = a[i] * b[i])
DEFINE_BINARY_FUNC(Div, c[i] = a[i] / b[i])



#endif //SIMPLE_CAFFE_UTILITY_HOST_ALTERNATE_HPP_
