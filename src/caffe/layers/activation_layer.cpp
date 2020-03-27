//
// Created by yipeng on 2020/3/25.
//
#include <vector>

#include "caffe/layers/activation_layer.hpp"

namespace caffe {

//子类重写虚函数 Reshape
template <typename Dtype>
void ActivationLayerInterface<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                              const vector<Tensor<Dtype>*>& top) {
	//输出tensor对象 reshape成与输入tensor同size
	top[0]->ReshapeLike(*bottom[0]);
}

//注册参数模板类
INSTANTIATE_CLASS(ActivationLayerInterface);
}        //namespace caffe