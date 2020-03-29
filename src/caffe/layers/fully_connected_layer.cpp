//
// Created by yipeng on 2020/3/26.
//
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fully_connected_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//重写 层初始化接口
template <typename Dtype>
void FullyConnectedLayer<Dtype>::LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
																					  const vector<Tensor<Dtype>*>& top) {
	//M是batch_size K是输入size N是输出size
	num_output_ = this->layer_param_.fully_connected_param().num_output();
	bias_term_ = this->layer_param_.fully_connected_param().bias_term();
	transpose_ = this->layer_param_.fully_connected_param().transpose();
	const int axis = bottom[0]->CanonicalAxisIndex(
		  this->layer_param_.fully_connected_param().axis());
	//如果bottom[0]的shape是N C H W, axis == 1 count就是C*H*W
	num_input_ = bottom[0]->count(axis);

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
			weight_shape[0] = num_input_;
			weight_shape[1] = num_output_;
		} else {
			weight_shape[0] = num_output_;
			weight_shape[1] = num_input_;
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
			vector<int> bias_shape(1, num_output_);
			this->weights_[1].reset(new Tensor<Dtype>(bias_shape));
			//fill填充偏置
			shared_ptr<FillerInterface<Dtype>> bias_filler(GetFiller<Dtype>(
						this->layer_param_.fully_connected_param().bias_filler()));
			bias_filler->Fill(this->weights_[1].get());
		}
	}
	//默认每个权重/偏置都计算梯度
	this->gradient_propagate_down_.resize(this->weights_.size(), true);
}

//重写 reshape接口
template <typename Dtype>
void FullyConnectedLayer<Dtype>::Reshape(const vector<Tensor<Dtype>*>& bottom,
                                         const vector<Tensor<Dtype>*>& top) {
	const int axis = bottom[0]->CanonicalAxisIndex(
		  this->layer_param_.fully_connected_param().axis());
	const int new_num_input = bottom[0]->count(axis);
	CHECK_EQ(num_input_, new_num_input) << "Input size incompatible with fully connected parameters";
	//如果axis == 1 得到的数量就是batch_size
	batch_size_ = bottom[0]->count(0, axis);

	vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(axis + 1);
	top_shape[axis] = num_output_;
	//输出size = batch_size * 输出数量 也就是M × N
	top[0]->Reshape(top_shape);

	//设置bias 初始化为1 size = batch_size
	if (bias_term_) {
		vector<int> bias_shape(1, batch_size_);
		bias_multiplier_.Reshape(bias_shape);
		caffe_set(batch_size_, Dtype(1), bias_multiplier_.mutable_cpu_data());
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
		                    batch_size_, num_output_, num_input_,
		                    Dtype(1),
		                    bottom_data,
		                    weight,
		                    Dtype(0),
		                    top_data);
	//输出结果上 再添加bias
	if (bias_term_) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
													batch_size_, num_output_, 1,
													Dtype(1),
													bias_multiplier_.cpu_data(),
													this->weights_[1]->cpu_data(),
													Dtype(1),
													top_data);
	}
}

//cpu fully connected backward pass
template <typename Dtype>
void FullyConnectedLayer<Dtype>::Backward_cpu(const vector<Tensor<Dtype>*>& top,
                                              const vector<bool>& error_propagate_down,
                                              const vector<Tensor<Dtype>*>& bottom) {
	//计算权重梯度
	if (this->gradient_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		//权重梯度 = 输入值的转置 .* 输出误差 (+ 权重梯度)
		if (transpose_) {
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
														num_input_, num_output_, batch_size_,
														Dtype(1),
														bottom_data,
														top_diff,
														Dtype(1),
														this->weights_[0]->mutable_cpu_diff());
		} else {
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				                    num_output_, num_input_, batch_size_,
				                    Dtype(1),
				                    top_diff,
				                    bottom_data,
				                    Dtype(1),
				                    this->weights_[0]->mutable_cpu_diff());
		}
	}

	//计算偏置梯度
	if (bias_term_ && this->gradient_propagate_down_[1]) {
		//偏置梯度 = 输出误差 (+ 偏置梯度)
		const Dtype* top_diff = top[0]->cpu_diff();
		caffe_cpu_gemv<Dtype>(CblasTrans,
													batch_size_, num_output_,
													Dtype(1),
													top_diff,
													bias_multiplier_.cpu_data(),
													Dtype(1),
													this->weights_[1]->mutable_cpu_diff());
	}

	//误差传递到下一层
	if (error_propagate_down[0]) {
		//输入diff = 权重的转置 .* 输出diff
		const Dtype* top_diff = top[0]->cpu_diff();
		if (transpose_) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
														batch_size_, num_input_, num_output_,
														Dtype(1),
														top_diff,
														this->weights_[0]->cpu_data(),
														Dtype(0),
														bottom[0]->mutable_cpu_diff());
		} else {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			                      batch_size_, num_input_, num_output_,
			                      Dtype(1),
			                      top_diff,
			                      this->weights_[0]->cpu_data(),
			                      Dtype(0),
			                      bottom[0]->mutable_cpu_diff());
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(FullyConnectedLayer);
#endif

//注册float double参数模板类
INSTANTIATE_CLASS(FullyConnectedLayer);
//注册全连接layer


}       //namespace caffe