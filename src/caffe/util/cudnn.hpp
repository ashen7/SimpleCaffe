//
// Created by yipeng on 2020/3/25.
//
#ifndef SIMPLE_CAFFE_UTILITY_CUDNN_HPP_
#define SIMPLE_CAFFE_UTILITY_CUDNN_HPP_

#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define CUDNN_VERSION_MIN(major, minor, patch) \
		(CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition) \
	do { \
		cudnnStatus_t status = condition; \
		CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " " << cudnnGetErrorString(status); \
	} while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
	switch (status) {
		case CUDNN_STATUS_SUCCESS:
			return "CUDNN_STATUS_SUCCESS";
		case CUDNN_STATUS_NOT_INITIALIZED:
			return "CUDNN_STATUS_NOT_INITIALIZED";
		case CUDNN_STATUS_ALLOC_FAILED:
			return "CUDNN_STATUS_ALLOC_FAILED";
		case CUDNN_STATUS_BAD_PARAM:
			return "CUDNN_STATUS_BAD_PARAM";
		case CUDNN_STATUS_INTERNAL_ERROR:
			return "CUDNN_STATUS_INTERNAL_ERROR";
		case CUDNN_STATUS_INVALID_VALUE:
			return "CUDNN_STATUS_INVALID_VALUE";
		case CUDNN_STATUS_ARCH_MISMATCH:
			return "CUDNN_STATUS_ARCH_MISMATCH";
		case CUDNN_STATUS_MAPPING_ERROR:
			return "CUDNN_STATUS_MAPPING_ERROR";
		case CUDNN_STATUS_EXECUTION_FAILED:
			return "CUDNN_STATUS_EXECUTION_FAILED";
		case CUDNN_STATUS_NOT_SUPPORTED:
			return "CUDNN_STATUS_NOT_SUPPORTED";
		case CUDNN_STATUS_LICENSE_ERROR:
			return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6, 0, 0)
		case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
			return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
		case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
			return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
		case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
			return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
#endif
	}

	return "Unknown cudnn status";
}

namespace caffe {
namespace cudnn {

template <typename Dtype> class dataType;
template <> class dataType<float> {
 public:
	static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
	static float zeroval;
	static float oneval;
	static const void* zero;
	static const void* one;
};
template <> class dataType<double> {
 public:
	static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
	static double zeroval;
	static double oneval;
	static const void* zero;
	static const void* one;
};

//创建cudnn 4d张量的描述符
template <typename Dtype>
inline void CreateTensor4dDesc(cudnnTensorDescriptor_t* tensor_desc) {
	CUDNN_CHECK(cudnnCreateTensorDescriptor(tensor_desc));
}

/*
 * 设置cudnn 4d张量的描述符
 * n: batch_size
 * c: channels
 * h: height
 * w: width
 */
template <typename Dtype>
inline void SetTensor4dDesc(cudnnTensorDescriptor_t* tensor_desc,
	                          int n, int c, int h, int w,
	                          int stride_n, int stride_c, int stride_h, int stride_w) {
	CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*tensor_desc,
		                                                dataType<Dtype>::type,
		                                                n, c, h, w,
		                                                stride_n, stride_c, stride_h, stride_w));
}

/*
 * 设置cudnn 4d张量的描述符
 * n: 输入/输出 batch_size
 * c: 输入/输出 channels
 * h: 输入/输出 height
 * w: 输入/输出 width
 */
template <typename Dtype>
inline void SetTensor4dDesc(cudnnTensorDescriptor_t* tensor_desc,
                            int n, int c, int h, int w) {
	const int stride_w = 1;
	const int stride_h = w * stride_w;
	const int stride_c = h * stride_h;
	const int stride_n = c * stride_c;
	SetTensor4dDesc<Dtype>(tensor_desc, n, c, h, w,
	                       stride_n, stride_c, stride_h, stride_w);
}

/*
 * 创建和设置cudnn 卷积核的描述符
 * n: 输出特征图个数/filter个数
 * c: 输入channels
 * h: filter height
 * w: filter width
 */
template <typename Dtype>
inline void CreateFilterDesc(cudnnFilterDescriptor_t* filter_desc,
                             int n, int c, int h, int w) {
	//创建卷积核的描述符
	CUDNN_CHECK(cudnnCreateFilterDescriptor(filter_desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
	//设置卷积核的描述符
	CUDNN_CHECK(cudnnSetFilter4dDescriptor(*filter_desc,
		                                              dataType<Dtype>::type,
		                                              CUDNN_TENSOR_NCHW,
		                                              n, c, h, w));
#else
	CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*filter_desc,
																					   dataType<Dtype>::type,
		                                         CUDNN_TENSOR_NCHW,
		                                         n, c, h, w));
#endif
}

//创建卷积操作的描述符
template <typename Dtype>
inline void CreateConvolutionDesc(cudnnConvolutionDescriptor_t* conv_desc) {
	CUDNN_CHECK(cudnnCreateConvolutionDescriptor((conv_desc)));
}

/*
 * 设置cudnn 卷积操作的描述符
 */
template <typename Dtype>
inline void SetConvolutionDesc(cudnnConvolutionDescriptor_t* conv_desc,
															 cudnnTensorDescriptor_t bottom,
															 cudnnFilterDescriptor_t filter,
															 int pad_h, int pad_w,
															 int stride_h, int stride_w) {
#if CUDNN_VERSION_MIN(6, 0, 0)
	CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv_desc,
																										   pad_h,
																										   pad_w,
																										   stride_h,
																										   stride_w,
																										   1,
																										   1,
																										   CUDNN_CROSS_CORRELATION,
																										   dataType<Dtype>::type));
#else
	CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv_desc,
																					    pad_h,
																					    pad_w,
																					    stride_h,
																					    stride_w,
																					    1,
																					    1,
																					    CUDNN_CROSS_CORRELATION));
#endif
}

/*
 * 创建和设置cudnn 池化操作的描述符
 */
template <typename Dtype>
inline void CreatePoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
	                            PoolingParameter_PoolMethod poolmethod,
	                            cudnnPoolingMode_t* mode,
	                            int h, int w,
	                            int pad_h, int pad_w,
	                            int stride_h, int stride_w) {
	switch (poolmethod) {
		case PoolingParameter_PoolMethod_MAX:
			*mode = CUDNN_POOLING_MAX;
			break;
		case PoolingParameter_PoolMethod_AVERAGE:
			*mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
			break;
		default:
			LOG(FATAL) << "Unknown pooling method";
	}
	//创建池化操作的描述符
	CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
	//设置池化操作的描述符
	CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc,
		                                              *mode,
																								  CUDNN_PROPAGATE_NAN,
																									h,
																									w,
																									pad_h,
																									pad_w,
																									stride_h,
																									stride_w));
#else
	CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(*pool_desc,
                                             *mode,
																					   CUDNN_PROPAGATE_NAN,
																						 h,
																						 w,
																						 pad_h,
																						 pad_w,
																						 stride_h,
																						 stride_w));
#endif
}

/*
 * 创建和设置cudnn 激活函数操作的描述符
 */
template <typename Dtype>
inline void CreateActivationDescriptor(cudnnActivationDescriptor_t* activation_desc,
																			 cudnnActivationMode_t mode) {

	//创建激活函数操作的描述符
	CUDNN_CHECK(cudnnCreateActivationDescriptor(activation_desc));
	//设置激活函数操作的描述符
	CUDNN_CHECK(cudnnSetActivationDescriptor(*activation_desc,
																										mode,
																										CUDNN_PROPAGATE_NAN,
																										Dtype(0)));
}

}      //namespace cudnn
}      //namespace caffe

#endif //USE_CUDNN
#endif //SIMPLE_CAFFE_UTILITY_CUDNN_HPP_
