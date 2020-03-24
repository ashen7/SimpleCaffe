//
// Created by yipeng on 2020/3/23.
//
#ifndef CAFFE_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_CAFFE_MAIN_HPP_

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"

//::func() 表示调用的是全局函数 而不是类中的函数 如果在类中调用同名函数 会调用全局的
namespace caffe {

//模板类型参数化
typedef ::testing::Types<float, double> TestDtypes;

//多gpu设备测试
template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
	typedef typename TypeParam::Dtype Dtype;
 protected:
	MultiDeviceTest() {
		Caffe::set_mode(TypeParam::device);
	}
	virtual ~MultiDeviceTest() {}
};

template <typename TypeParam>
struct CPUDevice {
	typedef TypeParam Dtype;
	static const Caffe::Mode device = Caffe::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype>> {
};

#ifdef CPU_ONLY
typedef ::testing::Types<CPUDevice<float>,
                         CPUDevice<double>> TestDtypesAndDevices;

#else
template <typename TypeParam>
struct GPUDevice {
typedef TypeParam Dtype;
static const Caffe::Mode device = Caffe::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype>> {
};

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                         CPUDevice<float>, CPUDevice<double>> TestDtypesAndDevices;
#endif

}      //namespace caffe

#endif //CAFFE_TEST_CAFFE_MAIN_HPP_
