//
// Created by yipeng on 2020/3/24.
//
#ifndef SIMPLE_CAFFE_LAYER_HPP_
#define SIMPLE_CAFFE_LAYER_HPP_

#include <vector>
#include <algorithm>
#include <string>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//层接口 多个层组成一个net 网络 层的子类必须实现前向计算 可以实现反向计算
template <typename Dtype>
class LayerInterface {
 public:
	//构造函数 传一个层参数 通过proto初始化权重和阶段
	explicit LayerInterface(const LayerParameter& param)
			: layer_param_(param) {
		phase_ = param.phase();
		if (layer_param_.tensors_size() > 0) {
			weights_.resize(layer_param_.tensors_size());
			for (int i = 0; i < layer_param_.tensors_size(); ++i) {
				//new 每个tensor对象 然后反序列化 reshape后分配内存 并得到数据 如果write_diff为ture 也会赋值diff
				weights_[i].reset(new Tensor<Dtype>());
				weights_[i]->FromProto(layer_param_.tensors(i));
			}
		}
	}
	virtual ~LayerInterface() {}

	/*
	 * 父类函数
	 * 公共层的初始化函数
	 * 参数: 预reshape的输入tensor
	 * 参数: 输出tensor 未reshape的同步内存对象
	 * 1. 检查输入和输出tensor是否正确
	 * 2. 调用层初始化为独立的层初始化 比如初始化cudnn
	 * 3. reshape为输出tensor所需要的分配空间
	 * 4. 为非零的损失权重设置损失权重
	 * */
	void SetUp(const vector<Tensor<Dtype>*>& bottom,
		         const vector<Tensor<Dtype>*>& top) {
		CheckTensorCounts(bottom, top);
		LayerSetUp(bottom, top);
		Reshape(bottom, top);
		SetLossWeights(top);
	}

	/*
	 * 虚函数 子类重写
	 * 独立层的初始化函数
	 * 参数: 预reshape的输入tensor data存放的这个层的输入数据
	 * 参数: 输出tensor 未reshape的同步内存对象
	 * 这个函数执行一次性的独立层初始化 比如初始化cudnn
	 * */
	virtual void LayerSetUp(const vector<Tensor<Dtype>*>& bottom,
	                        const vector<Tensor<Dtype>*>& top) {}

	/*
	 * 虚函数 子类重写
	 * 用输入tensor来调整输出tensor的shape
	 * 参数: 输入tensor 已经是输入的shape
	 * 参数: 输出tensor 调用reshape得到所需的size
	 */
	virtual void Reshape(const vector<Tensor<Dtype>*>& bottom,
	                     const vector<Tensor<Dtype>*>& top) = 0;

	/*
	 * 父类函数
	 * 给定输入tensor 计算输出tensor和loss
	 * 参数: 输入tensor data存放的本层的输入
	 * 参数: 输出tensor data存放的本层的输出
	 * 返回 本层的总loss
	 * 如果层有非零损失权重 计算返回loss
	 */
	inline Dtype Forward(const vector<Tensor<Dtype>*>& bottom,
	                     const vector<Tensor<Dtype>*>& top);

	/*
	 * 父类函数
	 * 给定输出层的误差梯度 计算输入层的误差梯度
	 * 参数: 输出tensor diff存放的是误差梯度
	 * 参数: 反向传播 一个和输入tensor list同size的bool list 每个索引表示对应tensor是否进行反向传播
	 * 参数: 输入tensor diff存放的是误差梯度
	 */
	inline void Backward(const vector<Tensor<Dtype>*>& top,
	                     const vector<bool>& propagate_down,
	                     const vector<Tensor<Dtype>*>& bottom);

	//得到权重
	vector<shared_ptr<Tensor<Dtype>>>& weights() {
		return weights_;
	}

	//得到层参数
	const LayerParameter& layer_param() const {
		return layer_param_;
	}

	//虚函数 子类重写 序列化 写层参数to proto
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

	//虚函数 子类重写 得到层类型
	virtual inline const char* type() const { return ""; }

	//虚函数 子类重写 层所需的输入tensor的个数
	virtual inline int NumBottomTensor() const { return -1; }
	//虚函数 子类重写 层所需的最小输入tensor的个数
	virtual inline int MinBottomTensor() const { return -1; }
	//虚函数 子类重写 层所需的最大输入tensor的个数
	virtual inline int MaxBottomTensor() const { return -1; }
	//虚函数 子类重写 层所需的输出tensor的个数
	virtual inline int NumTopTensor() const { return -1; }
	//虚函数 子类重写 层所需的最小输出tensor的个数
	virtual inline int MinTopTensor() const { return -1; }
	//虚函数 子类重写 层所需的最大输出tensor的个数
	virtual inline int MaxTopTensor() const { return -1; }
	//层要求输入和输出的tensor个数相同 返回true
	virtual inline bool EqualNumBottomTopTensor() const { return false; }

	//得到当前索引对应的tensor是否计算梯度
	inline bool param_propagate_down(const int param_index) {
		return (param_propagate_down_.size() > param_index) ?
				param_propagate_down_[param_index] : false;
	}

	//设置该索引对应的tensor是否计算梯度
	inline void set_param_propagate_down(const int param_index,
																		   const bool value) {
		if (param_propagate_down_.size() <= param_index) {
			param_propagate_down_.resize(param_index + 1, true);
		}
		param_propagate_down_[param_index] = value;
	}

 protected:
	LayerParameter layer_param_;                //层的参数
	Phase phase_;                               //train/test阶段
	vector<shared_ptr<Tensor<Dtype>>> weights_; //权重
	vector<bool> param_propagate_down_;         //和权重同size 有同样的tensor个数 表示每个tensor是否反向传播计算梯度
	vector<Dtype> loss_;                        //每个输出值是否有一个非零权重

	//内部函数 虚函数 子类重写 cpu forward pass接口
	virtual void Forward_cpu(const vector<Tensor<Dtype>*>& bottom,
	                         const vector<Tensor<Dtype>*>& top) = 0;
	//内部函数 虚函数 子类重写 gpu forward pass 如果没有重写gpu版本 后退调用cpu版本
	virtual void Forward_gpu(const vector<Tensor<Dtype>*>& bottom,
	                         const vector<Tensor<Dtype>*>& top) {
		return Forward_cpu(bottom, top);
	}

	//内部函数 虚函数 子类重写 cpu backward pass接口
	virtual void Backward_cpu(const vector<Tensor<Dtype>*>& top,
	                          const vector<bool>& propagate_down,
	                          const vector<Tensor<Dtype>*>& bottom) = 0;
	//内部函数 虚函数 子类重写 cpu backward pass 如果没有重写gpu版本 后退调用cpu版本
	virtual void Backward_gpu(const vector<Tensor<Dtype>*>& top,
	                          const vector<bool>& propagate_down,
	                          const vector<Tensor<Dtype>*>& bottom) {
		Backward_cpu(top, propagate_down, bottom);
	}

	//内部函数 父类函数 层初始化时调用 检查输入输出tensor是否正确
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
		//层是否要求输入和输出tensor个数相同
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

//父类函数 forward pass 包装了Forward_cpu/Forward_gpu实现
template <typename Dtype>
inline Dtype LayerInterface<Dtype>::Forward(const vector<Tensor<Dtype>*>& bottom,
                                            const vector<Tensor<Dtype>*>& top) {
	Dtype loss = 0;
	//子类重写的reshape 得到合适的输出tensor size
	Reshape(bottom, top);
	switch (Caffe::mode()) {
		case Caffe::CPU:
			//子类的前向计算 cpu实现
			Forward_cpu(bottom, top);
			for (int top_index = 0; top_index < top.size(); ++top_index) {
				//如果索引超出 或者loss在此索引的值是0 就不计算loss
				if (!this->loss(top_index)) {
					continue;
				}
				const int count = top[top_index]->count();
				const Dtype* data = top[top_index]->cpu_data();
				const Dtype* loss_weights = top[top_index]->cpu_diff();
				//loss加上输出data * diff的和
				loss += caffe_cpu_dot(count, data, loss_weights);
			}
			break;
		case Caffe::GPU:
			//子类的前向计算 gpu实现
			Forward_gpu(bottom, top);
#ifndef CPU_ONLY
			for (int top_index = 0; top_index < top.size(); ++top_index) {
				//如果索引超出 或者loss在此索引的值是0 就不计算loss
				if (!this->loss(top_index)) {
					continue;
				}
				const int count = top[top_index]->count();
				const Dtype *data = top[top_index]->gpu_data();
				const Dtype *loss_weights = top[top_index]->gpu_diff();
				Dtype tensor_loss = 0;
				//loss加上输出data * diff的和
				caffe_gpu_dot(count, data, loss_weights, &tensor_loss);
				loss += tensor_loss;
			}
#endif
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode";
	}

	return loss;
}

//父类函数 backward pass 包装了Backward_cpu/Backward_gpu实现
template <typename Dtype>
inline void Backward(const vector<Tensor<Dtype>*>& top,
                     const vector<bool>& propagate_down,
                     const vector<Tensor<Dtype>*>& bottom) {
	switch (Caffe::mode()) {
		case Caffe::CPU:
			//子类的反向计算 cpu实现
			Backward_cpu(top, propagate_down, bottom);
			break;
		case Caffe::GPU:
			//子类的反向计算 gpu实现
			Backward_gpu(top, propagate_down, bottom);
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode";
	}
}

//虚函数 序列化层参数 to google protocal buffer
template <typename Dtype>
void LayerInterface<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
	param->Clear();
	param->CopyFrom(layer_param_);
	param->clear_tensors();
	//proto反序列化的时候 得到了phase和权重 现在用新的权重序列化
	for (int i = 0; i < weights_.size(); ++i) {
		//每个tensor 会写入shape和data 如果write_diff为true 也会写入diff
		weights_[i]->ToProto(param->add_tensors(), write_diff);
	}
}

}      //namespace caffe

#endif //SIMPLE_CAFFE_LAYER_HPP_
