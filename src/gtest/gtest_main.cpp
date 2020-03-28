//
// Created by yipeng on 2020/3/23.
//
#include <gtest/gtest.h>

#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

//Google Test主入口
GTEST_API_ int main(int argc, char* argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	caffe::GlobalInit(&argc, &argv);
#ifndef CPU_ONLY
	caffe::Caffe::DeviceQuery();
#endif

	//run all test 返回这个值 测试原理是根据返回代码来判断测试是否成功
	return RUN_ALL_TESTS();
}
