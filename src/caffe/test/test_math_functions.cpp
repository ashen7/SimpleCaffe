//
// Created by yipeng on 2020/3/27.
//
#include <stdint.h>
#include <time.h>
#include <cmath>

#include <gtest/gtest.h>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

//测试固件类模板
template <typename TypeParam>
class MathFunctionsTest : public MultiDeviceTest<TypeParam> {

};


}      //namespace caffe