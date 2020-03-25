//
// Created by yipeng on 2020/3/24.
//
#ifndef SIMPLE_CAFFE_LAYER_HPP_
#define SIMPLE_CAFFE_LAYER_HPP_

#include <vector>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//层接口 多个层组成一个net 网络 层的子类必须实现前向计算 可以实现反向计算
template <typename Dtype>
class LayerInterface {
 public:
	//构造函数 传一个层参数 通过proto初始化权重
	explicit LayerInterface(const LayerParameter& param)
			: layer_param_(param) {
		phase_ = param.phase();
		if (layer_param_.tensor_list_size() > 0) {
			weights_.resize(layer_param_.tensor_list_size());
			for (int i = 0; i < layer_param_.tensor_list_size(); ++i) {
				weights_[i].reset(new Tensor<Dtype>());
				weights_[i]->FromProto(layer_param_.tensor_list(i));
			}
		}
	}
	virtual ~LayerInterface() {}

	/*
	 * 公共层的初始化函数
	 * 参数: 预reshape的输入tensor
	 * 参数: 输出tensor new了对象 但未reshape
	 * 1. 检查输入和输出tensor是否正确
	 * 2. 调用层初始化为独立的层初始化
	 * 3. reshape为输出tensor分配空间
	 * 4. 为非零的损失权重设置损失权重
	 * */
	void SetUp(const vector<Tensor<Dtype>*>& bottom,
		         const vector<Tensor<Dtype>*>& top) {

	}

	/*
	 * 独立层的初始化函数
	 * 参数: 预reshape的输入tensor data存放的这个层的输入数据
	 * 参数: 输出tensor new了对象 但未reshape
	 * 这个函数执行一次性的独立层初始化 from层的proto来初始化
	 * 设置输出tensor的reshape 分配空间 在这之前调用forward pass来调整输出tensor大小
	 * */
	virtual void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
	                        const vector<Tensor<Dtype>*>& top) {}

	/*
	 * 用输入tensor来调整输出tensor的shape
	 * 参数: 输入tensor 带要求的输入shape
	 * 参数: 输出tensor 按需调整shape
	 * 这个函数reshape输出tensor 为需要的shape
	 */
	virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
	                     const vector<Tensor<Dtype>*>& top) = 0;

	/*
	 * 给定输入tensor 计算输出tensor和loss
	 * 参数: 输入tensor data存放的本层的输入
	 * 参数: 输出tensor data存放的本层的输出
	 * 返回 本层的总loss
	 * 如果层有非零损失权重 计算返回loss
	 */
	inline Dtype Forward(const vector<Tensor<Dtype>*>& bottom,
	                     const vector<Tensor<Dtype>*>& top);

	/*
	 * 给定输出层的误差梯度 计算输入层的误差梯度
	 * 参数: 输出tensor diff存放的是误差梯度
	 * 参数: 反向传播 一个和输入tensor同shape的bool 表示每个索引是否进行反向传播
	 * 参数: 输入tensor diff存放的是误差梯度
	 *
	 */
	inline void Backward(const vector<Tensor<Dtype>*>& top,
	                     const vector<bool>& back_propagate,
	                     const vector<Tensor<Dtype>*>& bottom);

	//得到权重
	vector<shared_ptr<Tensor<Dtype>>>& weights() {
		return weights_;
	}

	//得到层参数
	const LayerParameter& layer_param() const {
		return layer_param_;
	}

	//写层参数to proto
	virtual void ToProto(LayerParameter* param, bool write_diff = false);

	//返回损失
	inline Dtype loss(const int top_index) const {
		return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
	}

	//设置损失
	inline void set_loss(const int top_index, const Dtype value) {
		if (loss_.size() <= top_index) {
			loss_.resize(top_index + 1, Dtype(0));
		}
		loss_[top_index] = value;
	}

	//得到层类型
	virtual inline const char* type() const { return ""; }

	//得到层所需的 输入tensor的大小
	virtual inline int NumBottomTensor() const { return -1; }

	virtual inline int MinBottomTensor() const { return -1; }

	virtual inline int MaxBottomTensor() const { return -1; }

	virtual inline int NumTopTensor() const { return -1; }

	virtual inline int MinTopTensor() const { return -1; }

	virtual inline int MaxTopTensor() const { return -1; }

	virtual inline bool EqualNumBottomTopTensor() const { return false; }

	inline bool param_back_propagate(const int param_index) {
		return (param_back_propagate_.size() > param_index) ?
				param_back_propagate_[param_index] : false;
	}

	//设置该索引是否计算梯度
	inline void set_param_back_propagate(const int param_index,
																		   const bool value) {
		if (param_back_propagate_.size() <= param_index) {
			param_back_propagate_.resize(param_index + 1, true);
		}
		param_back_propagate_[param_index] = value;
	}

 protected:
	LayerParameter layer_param_;                //层的参数
	Phase phase_;                               //train/test阶段
	vector<shared_ptr<Tensor<Dtype>>> weights_; //权重
	vector<bool> param_back_propagate_;         //是否反向传播
	vector<Dtype> loss_;                        //每个输出值是否有一个非零权重

	//cpu forward pass
	virtual void ForwardCPU(const vector<Tensor<Dtype>*>& bottom,
	                        const vector<Tensor<Dtype>*>& top) = 0;
	//gpu forward pass 如果没有gpu 后退用cpu
	virtual void ForwardGPU(const vector<Tensor<Dtype>*>& bottom,
	                        const vector<Tensor<Dtype>*>& top) {
		return ForwardCPU(bottom, top);
	}

	//cpu backward pass
	virtual void BackwardCPU(const vector<Tensor<Dtype>*>& top,
	                        onst vector<bool>& back_propagate,
	                        const vector<Tensor<Dtype>*>& bottom) = 0;
	//cpu backward pass
	virtual void BackwardGPU(const vector<Tensor<Dtype>*>& top,
	                        const vector<bool>& back_propagate,
	                        const vector<Tensor<Dtype>*>& bottom) {
		BackwardCPU(top, back_propagate, bottom);
	}

	//父类 层初始化时调用 检查输入输出tensor是否正确
	virtual void CheckTensorCounts(const vector<Tensor<Dtype>*>& bottom,
	                               const vector<Tensor<Dtype>*>& top) {
		if (NumBottomTensor() >= 0) {
			CHECK_EQ(NumBottomTensor(), bottom.size())
					<< type() << " Layer takes " << NumBottomTensor()
					<< " bottom tensor(s) as input";
		}
		if (MinBottomTensor() >= 0) {
			CHECK_LE(MinBottomTensor(), bottom.size())
					<< type() << " Layer takes at least " << MinBottomTensor()
					<< " bottom tensor(s) as input";
		}
		if (MaxBottomTensor() >= 0) {
			CHECK_GE(MaxBottomTensor(), bottom.size())
					<< type() << " Layer takes at most " << MaxBottomTensor()
					<< " bottom tensor(s) as input";
		}

		if (NumTopTensor() >= 0) {
			CHECK_EQ(NumTopTensor(), top.size())
					<< type() << " Layer produces " << NumTopTensor()
					<< " top tensor(s) as output";
		}
		if (MinTopTensor() >= 0) {
			CHECK_LE(MinTopTensor(), top.size())
					<< type() << " Layer produces at least " << MinTopTensor()
					<< " top tensor(s) as output";
		}
		if (MaxTopTensor() >= 0) {
			CHECK_GE(MaxTopTensor(), top.size())
					<< type() << " Layer produces at most " << MaxTopTensor()
					<< " top tensor(s) as output";
		}

		if (EqualNumBottomTopTensor()) {
			CHECK_EQ(bottom.size(), top.size())
					<< type() << " Layer produces one top tensor as output "
					<< "for each bottom tensor input";
		}
	}

	inline void SetLossWeights(const vector<Tensor<Dtype>*>& top) {

	}

 private:
	DISABLE_COPY_AND_ASSIGN(LayerInterface);
};     //class LayerInterface

//forward pass
template <typename Dtype>
inline Dtype LayerInterface<Dtype>::Forward(const vector<Tensor<Dtype>*>& bottom,
                                            const vector<Tensor<Dtype>*>& top) {
	Dtype loss = 0;
	Reshape(bottom, top);
	switch (Caffe::mode()) {
		case Caffe::CPU:
			ForwardCPU(bottom, top);
//			for ()
	}

}

}      //namespace caffe

#endif //SIMPLE_CAFFE_LAYER_HPP_
