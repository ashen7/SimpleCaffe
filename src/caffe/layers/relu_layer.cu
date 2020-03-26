//
// Created by yipeng on 2020/3/25.
//
#include <vector>
#include <algorithm>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {
//relu前向计算核函数
template <typename Dtype>
__global__ void ReLUForwardKernel(const int n, const Dtype* input,
																	Dtype* output, Dtype negative_slope) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		output[thread_id] = input[thread_id] > 0 ? input[thread_id] : negative_slope * input[thread_id];
	}
}

//gpu relu forward pass
template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
                                   const vector<Tensor<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
	ReLUForwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, top_data, negative_slope);
	CUDA_POST_KERNEL_CHECK;
}

//relu反向计算核函数
template <typename Dtype>
__global__ void ReLUBackwardKernel(const int n, const Dtype* input_diff,
	                                 const Dtype* input_data, Dtype* output_diff,
	                                 Dtype negative_slope) {
	CUDA_KERNEL_LOOP(thread_id, n) {
		output_diff[thread_id] = input_diff[thread_id] * ((input_data[thread_id] > 0) +
				negative_slope * (input_data[thread_id] <= 0));
	}
}

//gpu relu backward pass
template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Tensor<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Tensor<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
		ReLUBackwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, top_diff, bottom_data, bottom_diff, negative_slope);
		CUDA_POST_KERNEL_CHECK;
	}
}

//注册float double类型
INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);

}        //namespace caffe