//
// Created by yipeng on 2020/3/26.
//
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fully_connected_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FullyConnectedLayer<Dtype>::LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
																					  const vector<Tensor<Dtype>*>& top) {
	const int num_output = this->layer_param_.fully_connected_param().num_output();
	bias_term_ = this->layer_param_.fully_connected_param().bias_term();
	transpose_ = this->layer_param_.fully_connected_param().transpose();
	//M是batch_size K是输入size N是输出size
	N_ = num_output;
	const int axis = bottom[0]->CanonicalAxisIndex(
		  this->layer_param_.fully_connected_param().axis());
	//如果bottom[0]的shape是N C H W, axis == 1 count就是C*H*W
	K_ = bottom[0]->count(axis);

	if (this->weights_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (bias_term_) {
			//如果加偏置项
			this->weights_.resize(2);
		} else {
			this->weights_.resize(1);
		}
		//初始化权重
		vector<int> weight_shape(2);
		if (transpose_) {
			weight_shape[0] = K_;
			weight_shape[1] = N_;
		} else {
			weight_shape[0] = N_;
			weight_shape[1] = K_;
		}
		//给权重reshape 分配size = 输入数量 × 输出数量
		this->weights_[0].reset(new Tensor<Dtype>(weight_shape));
		//fill填充权重 填充类型由层参数的proto定义 返回一个对象指针 由shared_ptr保管
		shared_ptr<FillerInterface<Dtype>> weight_filler(GetFiller<Dtype>(
					this->layer_param_.fully_connected_param().weight_filler()));
		weight_filler->Fill(this->weights_[0].get());

		//如果添加bias为true 初始化偏置
		if (bias_term_) {
			//给权重reshape 分配size = 输出数量
			vector<int> bias_shape(1, N_);
			this->weights_[1].reset(new Tensor<Dtype>(bias_shape));
			//fill填充偏置
			shared_ptr<FillerInterface<Dtype>> bias_filler(GetFiller<Dtype>(
						this->layer_param_.fully_connected_param().bias_filler()));
			bias_filler->Fill(this->weights_[1].get());
		}
	}
	//默认权重的每个tensor(权重/偏置)都计算梯度
	this->param_propagate_down_.resize(this->weights_.size(), true);
}

//重写 reshape接口
template <typename Dtype>
void FullyConnectedLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                         const vector<Tensor<Dtype>*>& top) {
	const int axis = bottom[0]->CanonicalAxisIndex(
		  this->layer_param_.fully_connected_param().axis());
	const int new_K = bottom[0]->count(axis);
	CHECK_EQ(K_, new_K) << "Input size incompatible with fully connected parameters";
	//如果axis == 1 得到的数量就是batch_size
	M_ = bottom[0]->count(0, axis);

	vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(axis + 1);
	top_shape[axis] = N_;
	//输出size = batch_size * 输出数量 也就是M × N
	top[0]->Reshape(top_shape);

	//设置bias 初始化为1 size = batch_size
	if (bias_term_) {
		vector<int> bias_shape(1, M_);
		bias_multiplier_.Reshape(bias_shape);
		caffe_set(M_, Dtype(1), bias_multiplier_.mutalbe_cpu_data());
	}
}

//cpu fully connected forward pass
template <typename Dtype>
void FullyConnectedLayer<Dtype>::Forward_cpu(const vector<Tensor<Dtype>*>& bottom,
                                             const vector<Tensor<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* weight = this->weights_[0]->cpu_data();
	//矩阵相乘 输入(M*K) * 权重(K*N) = 输出(M*N)
	caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
		                    M_, N_, K_,
		                    Dtype(1),
		                    bottom_data,
		                    weight,
		                    Dtype(0),
		                    top_data);
	//输出结果上 再添加bias
	//输入: batch_size*1(value=1) 偏置: 1*输出size(value=bias) 输出: batch_size*输出size
	if (bias_term_) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
													M_, N_, 1,
													Dtype(1),
													bias_multiplier_.cpu_data(),
													this->weights_[1]->cpu_data(),
													Dtype(1),
													top_data);
	}
}

}       //namespace caffe