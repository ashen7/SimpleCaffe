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
	if (batch_size_ == 1) {
		//batch_size 为1 时 矩阵(权重)和向量(输入)相乘
		caffe_gpu_gemv<Dtype>(CblasNoTrans,
													num_output_, num_input_,
													Dtype(1),
													weight,
													bottom_data,
													Dtype(0),
													top_data);
		if (bias_term_) {
			//输出加上bias
			caffe_gpu_axpy<Dtype>(num_output_, bias_multiplier_.cpu_data()[0],
				                    this->weights_[1]->gpu_data(), top_data);
		}
	} else {
		//batch_size 不为1 矩阵(输入)和矩阵(权重)相乘
		caffe_gpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
													batch_size_, num_output_, num_input_,
													Dtype(1),
													bottom_data,
													weight,
													Dtype(0),
													top_data);
		if (bias_term_) {
			//输出加上bias
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
														batch_size_, num_output_, 1,
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
	//计算权重梯度
	if (this->gradient_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		//权重梯度 = 输入值的转置 .* 输出误差 (+ 权重梯度)
		if (transpose_) {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				                    num_input_, num_output_, batch_size_,
				                    Dtype(1),
				                    bottom_data,
				                    top_diff,
				                    Dtype(1),
				                    this->weights_[0]->mutable_gpu_diff());
		} else {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			                      num_output_, num_input_, batch_size_,
			                      Dtype(1),
			                      top_diff,
			                      bottom_data,
			                      Dtype(1),
			                      this->weights_[0]->mutable_gpu_diff());
		}
	}

	//计算偏置梯度
	if (bias_term_ && this->gradient_propagate_down_[1]) {
		//偏置梯度 = 输出误差 (+ 偏置梯度)
		const Dtype* top_diff = top[0]->gpu_diff();
		caffe_gpu_gemv<Dtype>(CblasTrans,
													batch_size_, num_output_,
													Dtype(1),
													top_diff,
													bias_multiplier_.gpu_data(),
			                    Dtype(1),
			                    this->weights_[1]->mutable_gpu_diff());
	}

	//误差传递到下一层
	if (error_propagate_down[0]) {
		//输入diff = 权重的转置 .* 输出diff
		const Dtype* top_diff = top[0]->gpu_diff();
		if (transpose_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
														batch_size_, num_input_, num_output_,
														Dtype(1),
														top_diff,
														this->weights_[0]->gpu_data(),
														Dtype(0),
														bottom[0]->mutable_gpu_diff());
		} else {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			                      batch_size_, num_input_, num_output_,
			                      Dtype(1),
			                      top_diff,
			                      this->weights_[0]->gpu_data(),
			                      Dtype(0),
			                      bottom[0]->mutable_gpu_diff());
		}
	}
}

//实例化gpu函数
INSTANTIATE_LAYER_GPU_FUNCS(FullyConnectedLayer);

}       //namespace caffe