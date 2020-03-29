//
// Created by yipeng on 2020/3/28.
//
#include <algorithm>
#include <vector>

#include <gtest/gtest.h>
#include <google/protobuf/text_format.h>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/fully_connected_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {




}      //namespace caffe