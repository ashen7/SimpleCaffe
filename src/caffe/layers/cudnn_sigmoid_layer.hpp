//
// Created by yipeng on 2020/3/25.
//
#ifndef SIMPLE_CAFFE_CUDNN_SIGMOID_LAYER_HPP_
#define SIMPLE_CAFFE_CUDNN_SIGMOID_LAYER_HPP_

#ifdef USE_CUDNN

#include <vector>

#include "caffe/tensor.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/activation_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
class CuDNNSigmoidLayer : public SigmoidLayer<Dtype> {
 public:
	explicit CuDNNSigmoidLayer(const LayerParameter& param)
			: SigmoidLayer<Dtype>(param), handles_setup_(false) {}

	//重写 层初始化接口
	virtual void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
													const vector<Tensor<Dtype>*>& top) override;
	//重写 reshape接口
	virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
	                     const vector<Tensor<Dtype>*>& top) override;
	virtual ~CuDNNSigmoidLayer();

 protected:
	/*
	 * 内部函数： 层的前向计算
	 * 输入tensor vector大小是1个tensor
	 * 输出tensor vector大小也是1个tensor
	 * 计算y = 1 / (1 + exp(-x))
	 */
	virtual void Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
	                         const vector<Tensor<Dtype>*>& top) override;
	/*
	 * 内部函数： 层的反向计算
	 * 输出tensor vector大小是1个tensor
	 * bool vector 表明下标对应值是否反向传播
	 * 输入tensor vector大小也是1个tensor
	 */
	virtual void Backward_gpu(const vector<Tensor<Dtype>*>& top,
	                          const vector<bool>& error_propagate_down,
	                          const vector<Tensor<Dtype>*>& bottom) override;

	bool handles_setup_;
	cudnnHandle_t cudnn_handle_;                  //cudnn句柄
	cudnnTensorDescriptor_t bottom_desc_;         //输入张量的描述符
	cudnnTensorDescriptor_t top_desc_;            //输出张量的描述符
	cudnnActivationDescriptor_t activation_desc_; //激活函数操作的描述符
};     //class CuDNNSigmoidLayer

}      //namespace caffe

#endif //USE_CUDNN
#endif //SIMPLE_CAFFE_CUDNN_SIGMOID_LAYER_HPP_
