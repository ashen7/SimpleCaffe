//
// Created by yipeng on 2020/3/26.
//
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fully_connected_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//gpu fully connected forward pass
template <typename Dtype>
void FullyConnectedLayer<Dtype>::Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
                                             const vector<Tensor<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* weight = this->weights_[0]->gpu_data();
	if (M_ == 1) {
		//batch_size 为1 时 矩阵(权重)和向量(输入)相乘
		caffe_gpu_gemv<Dtype>(CblasNoTrans,
													N_, K_,
													Dtype(1),
													weight,
													bottom_data,
													Dtype(0),
													top_data);
		if (bias_term_) {
			//输出加上bias
			caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
				                    this->weights_[1]->gpu_data(), top_data);
		}
	} else {
		//batch_size 不为1 矩阵(输入)和矩阵(权重)相乘
		caffe_gpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
													M_, N_, K_,
													Dtype(1),
													bottom_data,
													weight,
													Dtype(0),
													top_data);
		if (bias_term_) {
			//输出加上bias
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
														M_, N_, 1,
														Dtype(1),
														bias_multiplier_.gpu_data(),
														this->weights_[1]->gpu_data(),
														Dtype(1),
														top_data);
		}
	}
}

//gpu fully connected backward pass
template <typename Dtype>
void FullyConnectedLayer<Dtype>::Backward_gpu(const vector<Tensor<Dtype>*>& top,
                                              const vector<bool>& error_propagate_down,
                                              const vector<Tensor<Dtype>*>& bottom) {

}

//实例化gpu函数
INSTANTIATE_LAYER_GPU_FUNCS(FullyConnectedLayer);

}       //namespace caffe