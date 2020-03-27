//
// Created by yipeng on 2020/3/25.
//
#ifndef SIMPLE_CAFFE_SIGMOID_LAYER_HPP_
#define SIMPLE_CAFFE_SIGMOID_LAYER_HPP_

#include <vector>

#include "caffe/tensor.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/activation_layer.hpp"

namespace caffe {

//sigmoid激活函数
template <typename Dtype>
class SigmoidLayer : public ActivationLayerInterface<Dtype> {
 public:
	explicit SigmoidLayer(const LayerParameter& param)
			: ActivationLayerInterface<Dtype>(param) {}

	virtual ~SigmoidLayer() {}
	//层的类型
	virtual inline const char* type() const override { return "Sigmoid"; }

 protected:
	/*
	 * 内部函数： 层的前向计算
	 * 输入tensor vector大小是1个tensor
	 * 输出tensor vector大小也是1个tensor
	 * 计算y = 1 / (1 + exp(-x))
	 */
	virtual void Forward_cpu(const vector<Tensor<Dtype>*>& bottom,
	                         const vector<Tensor<Dtype>*>& top) override;
	virtual void Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
	                         const vector<Tensor<Dtype>*>& top) override;

	/*
	 * 内部函数： 层的反向计算
	 * 输出tensor vector大小是1个tensor
	 * bool vector 表明下标对应值是否反向传播
	 * 输入tensor vector大小也是1个tensor
	 */
	virtual void Backward_cpu(const vector<Tensor<Dtype>*>& top,
	                          const vector<bool>& error_propagate_down,
	                          const vector<Tensor<Dtype>*>& bottom) override;
	virtual void Backward_gpu(const vector<Tensor<Dtype>*>& top,
	                          const vector<bool>& error_propagate_down,
	                          const vector<Tensor<Dtype>*>& bottom) override;
};     //class SigmoidLayer

}      //namespace caffe

#endif //SIMPLE_CAFFE_SIGMOID_LAYER_HPP_
