//
// Created by yipeng on 2020/3/25.
//
#ifdef USE_CUDNN

#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

//cudnn relu前向计算
template <typename Dtype>
void CuDNNReLULayer<Dtype>::Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
                                        const vector<Tensor<Dtype>*>& top) {
	//如果设置了negative_slope的值 就后退到用CUDA计算 CUDNN不涉及这个参数
	if (ReLULayer<Dtype>::layer_param_.relu_param().negative_slope() != 0) {
		return ReLULayer<Dtype>::Forward_gpu(bottom, top);
	}

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
#if CUDNN_VERSION_MIN(5, 0, 0)
	CUDNN_CHECK(cudnnActivationForward(this->cudnn_handle_,
																							activation_desc_,
																							cudnn::dataType<Dtype>::one,
																							this->bottom_desc_,
																							bottom_data,
																							cudnn::dataType<Dtype>::zero,
																							this->top_desc_,
																							top_data));
#else
	CUDNN_CHECK(cudnnActivationForward_v4(this->cudnn_handle_,
																				activation_desc_,
																				cudnn::dataType<Dtype>::one,
																				this->bottom_desc_,
																				bottom_data,
																				cudnn::dataType<Dtype>::zero,
																				this->top_desc_,
																				top_data));
#endif
}

//cudnn relu反向计算
template <typename Dtype>
void CuDNNReLULayer<Dtype>::Backward_gpu(const vector<Tensor<Dtype>*>& top,
                                         const vector<bool>& error_propagate_down,
                                         const vector<Tensor<Dtype>*>& bottom) {
	//如果不误差传递 直接返回
	if (!error_propagate_down[0]) {
		return;
	}
	//如果设置了negative_slope的值 就后退到用CUDA计算 CUDNN不涉及这个参数
	if (ReLULayer<Dtype>::layer_param_.relu_param().negative_slope() != 0) {
		return ReLULayer<Dtype>::Backward_gpu(top, error_propagate_down, bottom);
	}

	const Dtype* top_data = top[0]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
#if CUDNN_VERSION_MIN(5, 0, 0)
	CUDNN_CHECK(cudnnActivationBackward(this->cudnn_handle_,
																							activation_desc_,
																							cudnn::dataType<Dtype>::one,
																							this->top_desc_,
																							top_data,
																							this->top_desc_,
																							top_diff,
																							this->bottom_desc_,
																							bottom_data,
																							cudnn::dataType<Dtype>::zero,
																							this->bottom_desc_,
																							bottom_diff));
#else
	CUDNN_CHECK(cudnnAactivationBackward_v4(this->cudnn_handle_,
																					activation_desc_,
																					cudnn::dataType<Dtype>::one,
																					this->top_desc_,
																					top_data,
																					this->top_desc_,
																					top_diff,
																					this->bottom_desc_,
																					bottom_data,
																					cudnn::dataType<Dtype>::zero,
																					this->bottom_desc_,
																					bottom_diff));
#endif
}

//实例化层的float和double类型 GPU前向计算和反向计算
INSTANTIATE_LAYER_GPU_FUNCS(CuDNNReLULayer);

}      //namespace caffe
#endif