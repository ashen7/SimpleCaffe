//
// Created by yipeng on 2020/3/23.
//
#include <gtest/gtest.h>

#include "caffe/common.hpp"
#include "caffe/synced_memory.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
//固件类fixture class 一个测试套件test suite里 多个测试test可以只用一次初始化数据
//使用SetUp()来初始化 TearDown()来析构
class CommonTest : public ::testing::Test {};

#ifndef CPU_ONLY  //GPU Caffe 单例测试

//使用夹具类 这里就用TEST_F 而不是TEST 两个参数 第一个是类名(test suite) 第二个是test name
TEST_F(CommonTest, TestCublas) {
	int device_id;
	CUDA_CHECK(cudaGetDevice(&device_id));
//	expect宏 test失败了产生非致命错误 不会退出
	EXPECT_TRUE(Caffe::cublas_handle());
}

#endif

TEST_F(CommonTest, TestMode) {
	Caffe::set_mode(Caffe::CPU);
	EXPECT_EQ(Caffe::mode(), Caffe::CPU);
	Caffe::set_mode(Caffe::GPU);
	EXPECT_EQ(Caffe::mode(), Caffe::GPU);
}

TEST_F(CommonTest, TestRandomSeedCPU) {
	SyncedMemory a(10 * sizeof(int));
	SyncedMemory b(10 * sizeof(int));
//	设置一样的seed 看看生成的随机值是否一样
	Caffe::set_random_seed(1701);
	caffe_rng_bernoulli(10, 0.5, static_cast<int*>(a.mutable_cpu_data()));

	Caffe::set_random_seed(1701);
	caffe_rng_bernoulli(10, 0.5, static_cast<int*>(b.mutable_cpu_data()));

	for (int i = 0; i < 10; ++i) {
		EXPECT_EQ(static_cast<const int*>(a.cpu_data())[i],
		          static_cast<const int*>(b.cpu_data())[i]);
	}
}

TEST_F(CommonTest, TestRandomSeedGPU) {
#ifndef CPU_ONLY
	SyncedMemory a(10 * sizeof(unsigned int));
	SyncedMemory b(10 * sizeof(unsigned int));
//	设置一样的seed 看看生成的随机值是否一样
	Caffe::set_random_seed(1701);
	CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
		           static_cast<unsigned int*>(a.mutable_gpu_data()), 10));

	Caffe::set_random_seed(1701);
	CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
		           static_cast<unsigned int*>(b.mutable_gpu_data()), 10));

	for (int i = 0; i < 10; ++i) {
		EXPECT_EQ(static_cast<const int*>(a.cpu_data())[i],
		          static_cast<const int*>(b.cpu_data())[i]);
	}
#endif
}

}       //namespace caffe
