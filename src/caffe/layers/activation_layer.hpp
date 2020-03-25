//
// Created by yipeng on 2020/3/25.
//
#ifndef SIMPLE_CAFFE_ACTIVATION_LAYER_HPP_
#define SIMPLE_CAFFE_ACTIVATION_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/tensor.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//激活函数层的接口类 输入和输出tensor同size
template <typename Dtype>
class ActivationLayerInterface : public LayerInterface<Dtype> {
 public:
	//子类构造调用父类构造
	explicit ActivationLayerInterface(const LayerParameter& param)
			: LayerInterface<Dtype>(param) {}
	virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
	                     const vector<Tensor<Dtype>*>& top) override;

	virtual inline int NumBottomTensor() const { return 1; }
	virtual inline int NumTopTensor() const { return 1; }
};     //class ActivationLayerInterface

}      //namespace caffe

#endif //SIMPLE_CAFFE_ACTIVATION_LAYER_HPP_
