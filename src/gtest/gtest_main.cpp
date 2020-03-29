//
// Created by yipeng on 2020/3/23.
//
#include <gtest/gtest.h>

#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
#ifndef CPU_ONLY
	cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif
}      //namespace caffe

#ifndef CPU_ONLY
	using caffe::CAFFE_TEST_CUDA_PROP;
#endif

//Google Test主入口
GTEST_API_ int main(int argc, char* argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	caffe::GlobalInit(&argc, &argv);

#ifndef CPU_ONLY
	caffe::Caffe::DeviceQuery();
	//给CAFFE_TEST_CUDA_PROP初始化
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device);
#endif

	//run all test 返回这个值 测试原理是根据返回代码来判断测试是否成功
	return RUN_ALL_TESTS();
}
