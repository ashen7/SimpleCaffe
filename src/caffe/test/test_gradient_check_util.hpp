//
// Created by yipeng on 2020/3/29.
//
#ifndef SIMPLE_CAFFE_TEST_GRADIENT_CHECK_UTIL_HPP_
#define SIMPLE_CAFFE_TEST_GRADIENT_CHECK_UTIL_HPP_

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <cmath>
#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"

namespace caffe {

//梯度检查
template <typename Dtype>
class GradientChecker {
 public:
	GradientChecker(const Dtype stepsize, const Dtype threshold,
		              const unsigned int seed = 1701, const Dtype kink = 0,
		              const Dtype kink_range = -1)
			: stepsize_(stepsize), threshold_(threshold), seed_(seed),
			  kink_(kink), kink_range_(kink_range) {}

	//检查梯度
	void CheckGradient(LayerInterface<Dtype>* layer,
										 const vector<Tensor<Dtype>*>& bottom,
										 const vector<Tensor<Dtype>*>& top,
										 int check_bottom = -1) {
		//层初始化
		layer->SetUp(bottom, top);
		CheckGradientSingle(layer, bottom, top, check_bottom, -1, -1);
	}
	//彻底检查梯度
	void CheckGradientExhaustive(LayerInterface<Dtype>* layer,
															 const vector<Tensor<Dtype>*>& bottom,
															 const vector<Tensor<Dtype>*>& top,
															 int check_bottom = -1);

	void CheckGradientElementWise(LayerInterface<Dtype>* layer,
																const vector<Tensor<Dtype>*>& bottom,
																const vector<Tensor<Dtype>*>& top);

	/*
	 * 检查单个输出的梯度
	 * check_bottom = i >= 0: 只检查它的输入tensor
	 * check_bottom == -1: 检查所有输入tensor和权重
	 * check_bottom < -1: 只检查权重
	 */
	void CheckGradientSingle(LayerInterface<Dtype>* layer,
	                         const vector<Tensor<Dtype>*>& bottom,
													 const vector<Tensor<Dtype>*>& top,
													 int check_bottom,
													 int top_index,
													 int top_data_index,
													 bool element_wise = false);

 protected:
	//得到loss
	Dtype GetObjAndGradient(const LayerInterface<Dtype>& layer,
	                        const vector<Tensor<Dtype>*>& top,
													int top_index = -1,
													int top_data_index = -1);
	Dtype stepsize_;    //更改权重时的一个很小的值
	Dtype threshold_;   //阈值
	unsigned int seed_; //随机数种子
	Dtype kink_;
	Dtype kink_range_;
};

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientElementWise(LayerInterface<Dtype>* layer,
                                                      const vector<Tensor<Dtype>*>& bottom,
																										  const vector<Tensor<Dtype>*>& top) {
	//层初始化 输出tensor分配空间
	layer->SetUp(bottom, top);
  CHECK_GT(top.size(), 0) << "ElementWise mode requires at least one top tensor";
	const int check_bottom = -1;
	const bool element_wise = true;
	//遍历输出tensor每一个输出值 梯度检查
	for (int i = 0; i < top.size(); ++i) {
		for (int j = 0; j < top[i]->count(); ++j) {
			CheckGradientSingle(layer, bottom, top, check_bottom, i, j, element_wise);
		}
	}
}

//彻底的梯度检查
template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientExhaustive(LayerInterface<Dtype>* layer,
                                                     const vector<Tensor<Dtype>*>& bottom,
                                                     const vector<Tensor<Dtype>*>& top,
                                                     int check_bottom) {
	//层初始化 输出tensor分配空间
	layer->SetUp(bottom, top);
	CHECK_GT(top.size(), 0) << "Exhaustive mode requires at least one top tensor";
	//遍历输出tensor每一个输出值 梯度检查
	for (int i = 0; i < top.size(); ++i) {
		for (int j = 0; j < top[i]->count(); ++j) {
			CheckGradientSingle(layer, bottom, top, check_bottom, i, j);
		}
	}
}

/*
  * 检查单个输出的梯度
	* check_bottom = i >= 0: 只检查它的输入tensor
	* check_bottom == -1: 检查所有输入tensor和权重
	* check_bottom < -1: 只检查权重
  *
*/
template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientSingle(LayerInterface<Dtype>* layer,
                                                 const vector<Tensor<Dtype>*>& bottom,
																								 const vector<Tensor<Dtype>*>& top,
																								 int check_bottom,
																								 int top_index,
																								 int top_data_index,
																								 bool element_wise) {
	//激活函数层
	if (element_wise) {
		CHECK_EQ(layer->weights().size(), 0);
		CHECK_LE(top_index, 0);
		CHECK_GE(top_data_index, 0);
		const int top_count = top[top_index]->count();
		for (int i = 0; i < bottom.size(); ++i) {
			CHECK_EQ(top_count, bottom[i]->count());
		}
	}

  //要检查的权重
	vector<Tensor<Dtype>*> tensors_to_check;
	vector<bool> propagate_down(bottom.size(), check_bottom == -1);
	//权重梯度和bias梯度设置为0 加入到要检查的张量中
	for (int i = 0; i < layer->weights().size(); ++i) {
		Tensor<Dtype>* weight = layer->weights()[i].get();
		caffe_set(weight->count(), Dtype(0), weight->mutable_cpu_diff());
		tensors_to_check.push_back(weight);
	}
	//输入tensor加入到要检查的张量中
	if (-1 == check_bottom) {
		for (int i = 0; i < bottom.size(); ++i) {
			tensors_to_check.push_back(bottom[i]);
		}
	} else if (check_bottom >= 0) {
		CHECK_LT(check_bottom, bottom.size());
		tensors_to_check.push_back(bottom[check_bottom]);
		propagate_down[check_bottom] = true;
	}

	CHECK_GT(tensors_to_check.size(), 0) << "No weights to check";
	Caffe::set_random_seed(seed_);
	//前向计算 得到输出值
	layer->Forward(bottom, top);
	//得到输出值对应索引的loss 也就是输出值×2 对应索引输出误差设置为2
	GetObjAndGradient(*layer, top, top_index, top_data_index);
	//计算梯度 并传递误差 梯度保存在weight的diff里 输入误差保存在bottom的diff里
	//每次调用只有索引对应的值是有输出误差值 = 2 其他的输出误差值均为0
	//而梯度和误差传递都是依靠输出误差值计算的
	layer->Backward(top, propagate_down, bottom);

	//存放所有要检查的张量(权重/输入) 的梯度/误差值
	vector<shared_ptr<Tensor<Dtype>>> computed_gradient_tensors(tensors_to_check.size());
	//将权重的梯度/输入的误差值 拷到computed_gradient_tensors的data里来
	for (int i = 0; i < tensors_to_check.size(); ++i) {
		Tensor<Dtype>* current_tensor = tensors_to_check[i];
		computed_gradient_tensors[i].reset(new Tensor<Dtype>());
		computed_gradient_tensors[i]->ReshapeLike(*current_tensor);
		const int count = tensors_to_check[i]->count();
		const Dtype* diff = tensors_to_check[i]->cpu_diff();
		Dtype* computed_gradients = computed_gradient_tensors[i]->mutable_cpu_data();
		caffe_copy(count, diff, computed_gradients);
	}

	//梯度检查: 每次改变一个点的权重 看看和ground truth的估计的梯度差距
	for (int i = 0; i < tensors_to_check.size(); ++i) {
		Tensor<Dtype>* current_tensor = tensors_to_check[i];
		Dtype* computed_gradients = computed_gradient_tensors[i]->mutable_cpu_data();

		for (int j = 0; j < current_tensor->count(); ++j) {
			Dtype estimated_gradient = 0;  //估计的梯度
			Dtype positive_objective = 0;  //客观正值
			Dtype negative_objective = 0;  //客观负值
			if (!element_wise || (j == top_data_index)) {
				//对应点的权重 + 一个很小的值
				current_tensor->mutable_cpu_data()[j] += stepsize_;
				Caffe::set_random_seed(seed_);
				//得到新的输出
				layer->Forward(bottom, top);
				//得到loss 新输出对应点*2
				positive_objective = GetObjAndGradient(*layer, top, top_index, top_data_index);

				//对应点的权重 - 一个很小的值
				current_tensor->mutable_cpu_data()[j] -= stepsize_ * 2;
				Caffe::set_random_seed(seed_);
				//得到新的输出
				layer->Forward(bottom, top);
				//得到loss 新输出对应点*2
				negative_objective = GetObjAndGradient(*layer, top, top_index, top_data_index);

				//将对应点权重保持原值
				current_tensor->mutable_cpu_data()[j] += stepsize_;
				//估计的梯度 = (pos梯度 - neg梯度) / +-时的小值 / 2
				estimated_gradient = (positive_objective - negative_objective) / stepsize_ / 2.0;
			}
			Dtype computed_gradient = computed_gradients[j];
			Dtype feature = current_tensor->cpu_data()[j];

			//kink: 0, kink_range: -1
			//索引对应的权重值/输入值 > -1 || < 1
			if (kink_ - kink_range_ > fabs(feature) ||
			    fabs(feature) > kink_ + kink_range_) {
				//真实梯度 和 实际梯度的最大值 要是小于1 就取1
				Dtype scale = std::max<Dtype>(std::max(fabs(computed_gradient),
					                                     fabs(estimated_gradient)),
					                            Dtype(1));
				//允许的误差 是阈值0.001
				EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
						<< "debug: (top_index, top_data_index, tensor_index, feat_index) = "
						<< top_index << ", " << top_data_index << ", " << i << ", " << j
						<< "; feat = " << feature
						<< "; objective+ = " << positive_objective
						<< "; objective- = " << negative_objective;
			}
		}
	}
}

//得到loss
template <typename Dtype>
Dtype GradientChecker<Dtype>::GetObjAndGradient(const LayerInterface<Dtype>& layer,
                                                const vector<Tensor<Dtype>*>& top,
																							  int top_index,
																							  int top_data_index) {
	Dtype loss = 0;
	if (top_index < 0) {
		//loss是所有tensor的输出值平方和的一半
		for (int i = 0; i < top.size(); ++i) {
			Tensor<Dtype>* top_tensor = top[i];
			const Dtype* top_tensor_data = top_tensor->cpu_data();
			Dtype* top_tensor_diff = top_tensor->mutable_cpu_diff();
			int count = top_tensor->count();
			for (int j = 0; j < count; ++j) {
				loss += top_tensor_data[j] * top_tensor_data[j];
			}
			//设置输出误差diff = 输出值data
			caffe_copy(top_tensor->count(), top_tensor_data, top_tensor_diff);
		}
		loss /= 2.0;
	} else {
		//1. 把输出tensor的误差值置0
		//2. 得到loss: 指定输出tensor指定索引的输出值 × 2
		//3. 设置输出tensor索引对应的误差值 = 2
		for (int i = 0; i < top.size(); ++i) {
			Tensor<Dtype>* top_tensor = top[i];
			Dtype* top_tensor_diff = top_tensor->mutable_cpu_diff();
			caffe_set(top_tensor->count(), Dtype(0), top_tensor_diff);
		}
		const Dtype loss_weight = 2;
		loss = top[top_index]->cpu_data()[top_data_index] * loss_weight;
		top[top_index]->mutable_cpu_diff()[top_data_index] = loss_weight;
	}

	return loss;
}

}     //namespace caffe


#endif //SIMPLE_CAFFE_TEST_GRADIENT_CHECK_UTIL_HPP_
