//
// Created by yipeng on 2020/3/25.
//
#include <cmath>

#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {
//sigmoid前向计算核函数
template <typename Dtype>
__global__ void SigmoidForwardKernel(const int n, const Dtype* input, Dtype* output) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		output[thread_id] = 0.5 * tanh(0.5 * input[thread_id]) + 0.5;
	}
}

//gpu sigmoid forward pass
template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
                                      const vector<Tensor<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	SigmoidForwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, top_data);
	CUDA_POST_KERNEL_CHECK;
}

//sigmoid反向计算核函数
template <typename Dtype>
__global__ void SigmoidBackwardKernel(const int n, const Dtype* input_diff,
	                                    const Dtype* output_data, Dtype* output_diff) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		const Dtype sigmoid_x = output_data[thread_id];
		output_diff[thread_id] = input_diff[thread_id] * sigmoid_x * (1.0 - sigmoid_x);
	}
}

//gpu sigmoid backward pass
template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_gpu(const vector<Tensor<Dtype>*>& top,
                                       const vector<bool>& error_propagate_down,
                                       const vector<Tensor<Dtype>*>& bottom) {
	//误差传递
	if (error_propagate_down[0]) {
		const Dtype* top_data = top[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		SigmoidBackwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, top_diff, top_data, bottom_diff);
		CUDA_POST_KERNEL_CHECK;
	}
}

//注册float double类型
INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);

}        //namespace caffe