//
// Created by yipeng on 2020/3/25.
//
#include <vector>

#include "caffe/layers/activation_layer.hpp"

namespace caffe {

//重写
template <typename Dtype>
void ActivationLayerInterface<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                              const vector<Tensor<Dtype>*>& top) {

}

}        //namespace caffe