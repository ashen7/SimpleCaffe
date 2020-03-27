//
// Created by yipeng on 2020/3/25.
//
#include <cmath>

#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

//sigmoid是0-1的区间 tanh是-1到1 乘以0.5就是-0.5到0.5 再加0.5就是0到1
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
	return 0.5 * tanh(0.5 * x) + 0.5;
}

//cpu sigmoid forward pass
template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Tensor<Dtype>*>& bottom,
                                      const vector<Tensor<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	for (int i = 0; i < count; ++i) {
		top_data[i] = sigmoid(bottom_data[i]);
	}
}

//cpu sigmoid backward pass
template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Tensor<Dtype>*>& top,
                                       const vector<bool>& error_propagate_down,
                                       const vector<Tensor<Dtype>*>& bottom) {
	//误差传递 输入diff = 输出diff * 输出值 * (1 - 输出值)
	if (error_propagate_down[0]) {
		const Dtype* top_data = top[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		for (int i = 0; i < count; ++i) {
			const Dtype sigmoid_x = top_data[i];
			bottom_diff[i] = top_diff[i] * sigmoid_x * (1.0 - sigmoid_x);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

//注册float double类型
INSTANTIATE_CLASS(SigmoidLayer);

}        //namespace caffe
