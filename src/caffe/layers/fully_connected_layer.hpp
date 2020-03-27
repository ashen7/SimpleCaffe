//
// Created by yipeng on 2020/3/26.
//
#ifndef SIMPLE_CAFFE_FULLY_CONNECTED_LAYER_HPP_
#define SIMPLE_CAFFE_FULLY_CONNECTED_LAYER_HPP_

#include <vector>

#include "caffe/tensor.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//全连接层 也可以叫Inner Product Layer
template <typename Dtype>
class FullyConnectedLayer : public LayerInterface<Dtype> {
 public:
	explicit FullyConnectedLayer(const LayerParameter& param)
			: LayerInterface<Dtype>(param) {}

	//重写 层初始化接口
	virtual void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
	                        const vector<Tensor<Dtype>*>& top) override;
	//重写 reshape接口
	virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
	                     const vector<Tensor<Dtype>*>& top) override;

	//层的类型
	virtual inline const char* type() const override { return "FullyConnected"; }
	virtual inline int NumBottomTensor() const { return 1; }
	virtual inline int NumTopTensor() const { return 1; }

 protected:
	/*
	 * 内部函数： 层的前向计算
	 * 输入tensor vector大小是1个tensor
	 * 输出tensor vector大小也是1个tensor
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

	int M_;           //batch size
	int N_;           //输出size
	int K_;           //输入size
	bool bias_term_;  //是否添加偏置项
	Tensor<Dtype> bias_multiplier_;  //值为1(相当于值为1的输入) size = batch_size
	bool transpose_;  //权重是否转置
};     //class FullyConnectedLayer

}      //namespace caffe

#endif //SIMPLE_CAFFE_FULLY_CONNECTED_LAYER_HPP_
