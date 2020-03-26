//
// Created by yipeng on 2020/3/25.
//
#ifdef USE_CUDNN

#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

//重写 层初始化接口
template <typename Dtype>
void CuDNNReLULayer<Dtype>::LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
																			 const vector<Tensor<Dtype>*>& top) {
	//父类神经层接口类和relu层类都没有重写 层初始化接口 调用的是父类layer的层初始化 什么都不做
	ReLULayer<Dtype>::LayerSetUp(bottom, top);
	//初始化 cuDNN
	CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
	cudnn::CreateTensor4dDesc<Dtype>(&bottom_desc_);
	cudnn::CreateTensor4dDesc<Dtype>(&top_desc_);
	cudnn::CreateActivationDescriptor<Dtype>(&activation_desc_, CUDNN_ACTIVATION_RELU); //mode是ReLU
	handles_setup_ = true;
}

//重写 reshape接口
template <typename Dtype>
void CuDNNReLULayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                    const vector<Tensor<Dtype>*>& top) {
	//调用的父类神经层接口类的reshape 输出和输入tensor同size
	ReLULayer<Dtype>::Reshape(bottom, top);
	const int N = bottom[0]->num();
	const int C = bottom[0]->channels();
	const int H = bottom[0]->height();
	const int W = bottom[0]->width();
	//设置输入 张量描述符
	cudnn::SetTensor4dDesc<Dtype>(&bottom_desc_, N, C, H, W);
	//设置输出 张量描述符
	cudnn::SetTensor4dDesc<Dtype>(&top_desc_, N, C, H, W);
}

//虚析构 摧毁handle和描述符 调用完子类 会调用父类析构
template <typename Dtype>
CuDNNReLULayer<Dtype>::~CuDNNReLULayer() {
	if (!handles_setup_) {
		return ;
	}
	cudnnDestroyTensorDescriptor(this->bottom_desc_);
	cudnnDestroyTensorDescriptor(this->top_desc_);
	cudnnDestroyActivationDescriptor(this->activation_desc_);
	cudnnDestroy(this->cudnn_handle_);
}

//注册float double类型
INSTANTIATE_CLASS(CuDNNReLULayer);

}       //namespace caffe
#endif  //USE_CUDNN
