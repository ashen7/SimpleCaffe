//
// Created by yipeng on 2020/3/25.
//
#include <vector>
#include <algorithm>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

//cpu relu forward pass
template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Tensor<Dtype>*>& bottom,
                                   const vector<Tensor<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
	for (int i = 0; i < count; ++i) {
		top_data[i] = std::max(bottom_data[i], Dtype(0)) +
				negative_slope * std::min(bottom_data[i], Dtype(0));
	}
}

//cpu relu backward pass
template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Tensor<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Tensor<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
		for (int i = 0; i < count; ++i) {
			bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0) +
					negative_slope * (bottom_data[i] <= 0));
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

//注册float double类型
INSTANTIATE_CLASS(ReLULayer);

}        //namespace caffe
