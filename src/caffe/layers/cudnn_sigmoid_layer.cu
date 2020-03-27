//
// Created by yipeng on 2020/3/25.
//
#ifdef USE_CUDNN

#include <vector>

#include "caffe/layers/cudnn_sigmoid_layer.hpp"

namespace caffe {

//cudnn sigmoid前向计算
template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
                                           const vector<Tensor<Dtype>*>& top) {
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

//cudnn sigmoid反向计算
template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Backward_gpu(const vector<Tensor<Dtype>*>& top,
                                            const vector<bool>& error_propagate_down,
                                            const vector<Tensor<Dtype>*>& bottom) {
	//如果不误差传递 直接返回
	if (!error_propagate_down[0]) {
		return;
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
INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSigmoidLayer);

}      //namespace caffe
#endif //USE_CUDNN