//
// Created by yipeng on 2020/3/23.
//
#ifndef SIMPLE_CAFFE_FILL_HPP_
#define SIMPLE_CAFFE_FILL_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/tensor.hpp"
#include "caffe/synced_memory.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//用随机数或常数 来填充 tensor的抽象(接口)类
template <typename Dtype>
class FillerInterface {
 public:
	explicit FillerInterface(const FillerParameter& param) : fill_param_(param) {}
	virtual ~FillerInterface() {}
	virtual void Fill(Tensor<Dtype>* tensor) = 0;  //纯虚函数 接口
 protected:
	//protected定义的成员 通过public继承 也是protected成员
	FillerParameter fill_param_;
};     //class FilterInterface

//常数填充
template <typename Dtype>
class ConstantFiller : public FillerInterface<Dtype> {
 public:
	//子类构造 调用父类构造
	explicit ConstantFiller(const FillerParameter& param)
			: FillerInterface<Dtype>(param) {}

	virtual void Fill(Tensor<Dtype>* tensor) override {
		Dtype* data = tensor->mutable_cpu_data();
		const int count = tensor->count();
		const Dtype value = this->fill_param_.value();
		CHECK_GT(count, 0);
		for (int i = 0; i < count; ++i) {
			data[i] = value;
		}
	}
};     //class ConstantFill

//生成均匀分布的随机数填充
template <typename Dtype>
class UniformFiller : public FillerInterface<Dtype> {
 public:
	//子类构造 调用父类构造
	explicit UniformFiller(const FillerParameter& param)
		: FillerInterface<Dtype>(param) {}

	virtual void Fill(Tensor<Dtype>* tensor) override {
		CHECK_GT(tensor->count(), 0);
		caffe_rng_uniform<Dtype>(tensor->count(), Dtype(this->fill_param_.min()),
			                       Dtype(this->fill_param_.max()), tensor->mutable_cpu_data());
	}
};     //class UniformFill

//生成高斯分布/正态分布的随机数填充
template <typename Dtype>
class GaussianFiller : public FillerInterface<Dtype> {
public:
	//子类构造 调用父类构造
	explicit GaussianFiller(const FillerParameter& param)
		: FillerInterface<Dtype>(param) {}

	virtual void Fill(Tensor<Dtype>* tensor) override {
		Dtype* data = tensor->mutable_cpu_data();
		CHECK_GT(tensor->count(), 0);
		caffe_rng_gaussian(tensor->count(), Dtype(this->fill_param_.mean()),
		                   Dtype(this->fill_param_.stddev()), tensor->mutable_cpu_data());
		int sparse = this->fill_param_.sparse();
	}
};     //class GaussianFill

//Xavier初始化 生成特定值范围的均匀分布的随机数填充
template <typename Dtype>
class XavierFiller : public FillerInterface<Dtype> {
public:
	//子类构造 调用父类构造
	explicit XavierFiller(const FillerParameter& param)
		: FillerInterface<Dtype>(param) {}

	virtual void Fill(Tensor<Dtype>* tensor) override {
		CHECK_GT(tensor->count(), 0);
		//C * H * W
		int fan_in = tensor->count() / tensor->shape(0);
		//N * H * W
		int fan_out = tensor->num_axes() > 1 ?
									tensor->count() / tensor->shape(1) :
									tensor->count();
		Dtype n = fan_in;  //默认fan_in
		if (this->fill_param_.variance_norm() ==
		    FillerParameter_VarianceNorm_AVERAGE) {
			n = (fan_in + fan_out) / Dtype(2);
		} else if (this->fill_param_.variance_norm() ==
		    FillerParameter_VarianceNorm_FAN_OUT) {
			n = fan_out;
		}
		//范围scale 是3/n的开方
		Dtype scale = sqrt(Dtype(3) / n);
		caffe_rng_uniform<Dtype>(tensor->count(), -scale, scale,
		                         tensor->mutable_cpu_data());
	}
};     //class XavierFiller

//通过filler参数的type 来得到一个filler多态对象 父类指针指向子类对象
template <typename Dtype>
FillerInterface<Dtype>* GetFiller(const FillerParameter& param) {
	const std::string& type = param.type();
	if (type == "constant") {
		return new ConstantFiller<Dtype>(param);
	} else if (type == "uniform") {
		return new UniformFiller<Dtype>(param);
	} else if (type == "gaussian") {
		return new GaussianFiller<Dtype>(param);
	} else if (type == "xavier") {
		return new XavierFiller<Dtype>(param);
	} else {
		CHECK(false) << "Unknown filler name: " << type;
	}
	return (FillerInterface<Dtype>*)(nullptr);
}

}      //namespace caffe

#endif //SIMPLE_CAFFE_FILL_HPP_
