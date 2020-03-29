//
// Created by yipeng on 2020/3/28.
//
#include <vector>

#include <gtest/gtest.h>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/fully_connected_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

//测试固件类模板
template <typename TypeParam>
class FullyConnectedLayerTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;
 protected:
	FullyConnectedLayerTest()
			: bottom_tensor_(new Tensor<Dtype>(2, 3, 4, 5)),
			  bottom_nobatch_tensor_(new Tensor<Dtype>(1, 2, 3, 4)),
			  top_tensor_(new Tensor<Dtype>()) {
		//填充值
		FillerParameter filler_param;
		filler_param.set_min(Dtype(0));
		filler_param.set_max(Dtype(1));
		UniformFiller<Dtype> filler(filler_param);
		filler.Fill(this->bottom_tensor_);

		top_tensor_list_.push_back(top_tensor_);
		bottom_tensor_list_.push_back(bottom_tensor_);

		fully_connected_param_ = layer_param_.mutable_fully_connected_param();
		//设置层的全连接层参数
		fully_connected_param_->set_bias_term(true);
		fully_connected_param_->set_axis(1);
		fully_connected_param_->mutable_weight_filler()->set_type("constant");
		fully_connected_param_->mutable_weight_filler()->set_value(0);
		fully_connected_param_->mutable_bias_filler()->set_type("constant");
		fully_connected_param_->mutable_bias_filler()->set_value(0);
	}

	~FullyConnectedLayerTest() {
		if (bottom_tensor_) {
			delete bottom_tensor_;
		}
		if (bottom_nobatch_tensor_) {
			delete bottom_nobatch_tensor_;
		}
		if (top_tensor_) {
			delete top_tensor_;
		}
	}

	Tensor<Dtype>* const bottom_tensor_;
	Tensor<Dtype>* const bottom_nobatch_tensor_;
	Tensor<Dtype>* const top_tensor_;
	vector<Tensor<Dtype>*> bottom_tensor_list_;
	vector<Tensor<Dtype>*> top_tensor_list_;
	LayerParameter layer_param_;
	FullyConnectedParameter* fully_connected_param_;
};

//注册测试用例 模板类型参数化 测试CPU和GPU的float double两个类型
TYPED_TEST_CASE(FullyConnectedLayerTest, TestDtypesAndDevices);

TYPED_TEST(FullyConnectedLayerTest, TestSetUp) {
	typedef typename TypeParam::Dtype Dtype;
	//设置层的全连接层参数
	this->fully_connected_param_->set_num_output(10);

	shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
	//层初始化
	layer->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	EXPECT_EQ(this->top_tensor_->num(), 2);
	EXPECT_EQ(this->top_tensor_->channels(), 10);
	EXPECT_EQ(this->top_tensor_->height(), 1);
	EXPECT_EQ(this->top_tensor_->width(), 1);
}

TYPED_TEST(FullyConnectedLayerTest, TestSetUpTransposeFalse) {
	typedef typename TypeParam::Dtype Dtype;
	//设置层的全连接层参数
	this->fully_connected_param_->set_num_output(10);
	this->fully_connected_param_->set_transpose(false);

	shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
	//层初始化
	layer->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	EXPECT_EQ(this->top_tensor_->num(), 2);
	EXPECT_EQ(this->top_tensor_->channels(), 10);
	EXPECT_EQ(this->top_tensor_->height(), 1);
	EXPECT_EQ(this->top_tensor_->width(), 1);
	EXPECT_EQ(layer->weights()[0]->num_axes(), 2);
	//权重不转置 就是num_output * num_input
	EXPECT_EQ(layer->weights()[0]->shape(0), 10);
	EXPECT_EQ(layer->weights()[0]->shape(1), 60);
}

TYPED_TEST(FullyConnectedLayerTest, TestSetUpTransposeTrue) {
	typedef typename TypeParam::Dtype Dtype;
	//设置层的全连接层参数
	this->fully_connected_param_->set_num_output(10);
	this->fully_connected_param_->set_transpose(true);

	shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
	//层初始化
	layer->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	EXPECT_EQ(this->top_tensor_->num(), 2);
	EXPECT_EQ(this->top_tensor_->channels(), 10);
	EXPECT_EQ(this->top_tensor_->height(), 1);
	EXPECT_EQ(this->top_tensor_->width(), 1);
	EXPECT_EQ(layer->weights()[0]->num_axes(), 2);
	//权重转置 num_input * num_output
	EXPECT_EQ(layer->weights()[0]->shape(0), 60);
	EXPECT_EQ(layer->weights()[0]->shape(1), 10);
}

TYPED_TEST(FullyConnectedLayerTest, TestForward) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
	    sizeof(Dtype) == 4 ||
	    IS_VALID_CUDA) {
		//设置层的全连接层参数
		this->fully_connected_param_->set_num_output(10);
		this->fully_connected_param_->mutable_weight_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_weight_filler()->set_min(0);
		this->fully_connected_param_->mutable_weight_filler()->set_max(1);
		this->fully_connected_param_->mutable_bias_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_bias_filler()->set_min(1);
		this->fully_connected_param_->mutable_bias_filler()->set_max(2);

		shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
		//层初始化
		layer->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
		layer->Forward(this->bottom_tensor_list_, this->top_tensor_list_);
		const int count = this->top_tensor_->count();
		const Dtype* data = this->top_tensor_->cpu_data();
		//因为bias大于1 输入为正0-1 权重为正0-1 所以结果一定大于1
		for (int i = 0; i < count; ++i) {
			EXPECT_GE(data[i], 1);
		}
	} else {
		LOG(ERROR) << "Skipping test due to old architecture";
	}
}

TYPED_TEST(FullyConnectedLayerTest, TestForwardTranspose) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
	    sizeof(Dtype) == 4 ||
	    IS_VALID_CUDA) {
		//设置层的全连接层参数
		this->fully_connected_param_->set_num_output(10);
		this->fully_connected_param_->mutable_weight_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_weight_filler()->set_min(0);
		this->fully_connected_param_->mutable_weight_filler()->set_max(1);
		this->fully_connected_param_->mutable_bias_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_bias_filler()->set_min(1);
		this->fully_connected_param_->mutable_bias_filler()->set_max(2);
		this->fully_connected_param_->set_transpose(false);

		shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
		//层初始化
		layer->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
		layer->Forward(this->bottom_tensor_list_, this->top_tensor_list_);

		//存放不转置权重的输出tensor值
		const int count = this->top_tensor_->count();
		shared_ptr<Tensor<Dtype>> top = std::make_shared<Tensor<Dtype>>();
		top->ReshapeLike(*this->top_tensor_);
		caffe_copy(count, this->top_tensor_->cpu_data(), top->mutable_cpu_data());

		//转置权重
		this->top_tensor_list_.clear();
		this->top_tensor_list_.push_back(new Tensor<Dtype>());
		this->fully_connected_param_->set_transpose(true);
		shared_ptr<FullyConnectedLayer<Dtype>> layer_transpose(new FullyConnectedLayer<Dtype>(this->layer_param_));
		layer_transpose->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
		//权重的数量
		const int weight_count = layer->weights()[0]->count(); //600
		EXPECT_EQ(weight_count, layer_transpose->weights()[0]->count()); //600

		const Dtype* weight = layer->weights()[0]->cpu_data(); //10 x 60
		Dtype* weight_transpose = layer_transpose->weights()[0]->mutable_cpu_data(); // 60 x 10
		const int width = layer->weights()[0]->shape(1);  //60
		const int width_transpose = layer_transpose->weights()[0]->shape(1); //10
		//不转置权重 拷给 转置的权重
		for (int i = 0; i < weight_count; ++i) {
			int row = i / width;  //0-9
			int col = i % width;  //0-59
			//把右边每一行 (移动)列的值 赋给左边每一列 (移动)行
			weight_transpose[col * width_transpose + row] = weight[row * width + col];
		}
		//层的bias拷给层转置的bias
		ASSERT_EQ(layer->weights()[1]->count(), layer_transpose->weights()[1]->count());
		caffe_copy(layer->weights()[1]->count(), layer->weights()[1]->cpu_data(),
			         layer_transpose->weights()[1]->mutable_cpu_data());

		//保存权重和bias一样 再进行前向计算
		layer_transpose->Forward(this->bottom_tensor_list_, this->top_tensor_list_);
		EXPECT_EQ(count, this->top_tensor_->count())
				<< "Invalid count for top tensor for Fully Connected with transpose";

		//存放转置权重的输出tensor值
		shared_ptr<Tensor<Dtype>> top_transpose = std::make_shared<Tensor<Dtype>>();
		top_transpose->ReshapeLike(*this->top_tensor_list_[0]);
		caffe_copy(count, this->top_tensor_list_[0]->cpu_data(), top_transpose->mutable_cpu_data());
		const Dtype* data = top->cpu_data();
		const Dtype* data_transpose = top_transpose->cpu_data();
		//比较输出值
		for (int i = 0; i < count; ++i) {
			EXPECT_FLOAT_EQ(data[i], data_transpose[i]);
		}
	} else {
		LOG(ERROR) << "Skipping test due to old architecture";
	}
}

TYPED_TEST(FullyConnectedLayerTest, TestForwardNoBatch) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
	    sizeof(Dtype) == 4 ||
	    IS_VALID_CUDA) {
		//设置层的全连接层参数
		this->fully_connected_param_->set_num_output(10);
		this->fully_connected_param_->mutable_weight_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_weight_filler()->set_min(0);
		this->fully_connected_param_->mutable_weight_filler()->set_max(1);
		this->fully_connected_param_->mutable_bias_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_bias_filler()->set_min(1);
		this->fully_connected_param_->mutable_bias_filler()->set_max(2);

		shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
		//层初始化
		layer->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
		layer->Forward(this->bottom_tensor_list_, this->top_tensor_list_);

		const int count = this->top_tensor_->count();
		const Dtype* data = this->top_tensor_->cpu_data();
		for (int i = 0; i < count; ++i) {
			EXPECT_GE(data[i], 1);
		}
	} else {
		LOG(ERROR) << "Skipping test due to old architecture";
	}
}

TYPED_TEST(FullyConnectedLayerTest, TestGradient) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
	    sizeof(Dtype) == 4 ||
	    IS_VALID_CUDA) {
		//设置层的全连接层参数
		this->fully_connected_param_->set_num_output(10);
		this->fully_connected_param_->mutable_weight_filler()->set_type("gaussian");
		this->fully_connected_param_->mutable_weight_filler()->set_mean(0);
		this->fully_connected_param_->mutable_weight_filler()->set_stddev(1);
		this->fully_connected_param_->mutable_bias_filler()->set_type("gaussian");
		this->fully_connected_param_->mutable_bias_filler()->set_mean(0);
		this->fully_connected_param_->mutable_bias_filler()->set_stddev(1);

		shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
		//梯度检查 参数分别是梯度检查时改变权重的那个很小的值 和 梯度真实值 实际值差距的阈值
		GradientChecker<Dtype> gradient_checker(1e-2, 1e-3);
		gradient_checker.CheckGradientExhaustive(layer.get(), this->bottom_tensor_list_, this->top_tensor_list_);
	} else {
		LOG(ERROR) << "Skipping test due to old architecture";
	}
}

TYPED_TEST(FullyConnectedLayerTest, TestGradientTranspose) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
	    sizeof(Dtype) == 4 ||
	    IS_VALID_CUDA) {
		//设置层的全连接层参数
		this->fully_connected_param_->set_num_output(10);
		this->fully_connected_param_->mutable_weight_filler()->set_type("gaussian");
		this->fully_connected_param_->mutable_weight_filler()->set_mean(0);
		this->fully_connected_param_->mutable_weight_filler()->set_stddev(1);
		this->fully_connected_param_->mutable_bias_filler()->set_type("gaussian");
		this->fully_connected_param_->mutable_bias_filler()->set_mean(0);
		this->fully_connected_param_->mutable_bias_filler()->set_stddev(1);
		//权重转置
		this->fully_connected_param_->set_transpose(true);

		shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
		//梯度检查 参数分别是梯度检查时改变权重的那个很小的值 和 梯度真实值 实际值差距的阈值
		GradientChecker<Dtype> gradient_checker(1e-2, 1e-3);
		gradient_checker.CheckGradientExhaustive(layer.get(), this->bottom_tensor_list_, this->top_tensor_list_);
	} else {
		LOG(ERROR) << "Skipping test due to old architecture";
	}
}

TYPED_TEST(FullyConnectedLayerTest, TestBackwardTranspose) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
	    sizeof(Dtype) == 4 ||
	    IS_VALID_CUDA) {
		//设置层的全连接层参数
		this->fully_connected_param_->set_num_output(10);
		this->fully_connected_param_->mutable_weight_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_weight_filler()->set_min(0);
		this->fully_connected_param_->mutable_weight_filler()->set_max(1);
		this->fully_connected_param_->mutable_bias_filler()->set_type("uniform");
		this->fully_connected_param_->mutable_bias_filler()->set_min(1);
		this->fully_connected_param_->mutable_bias_filler()->set_max(2);

		shared_ptr<FullyConnectedLayer<Dtype>> layer(new FullyConnectedLayer<Dtype>(this->layer_param_));
		//层初始化
		layer->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
		layer->Forward(this->bottom_tensor_list_, this->top_tensor_list_);

		//copy 输出tensor data
		shared_ptr<Tensor<Dtype>> top = std::make_shared<Tensor<Dtype>>();
		top->CopyFrom(*this->top_tensor_, false, true);
		shared_ptr<Tensor<Dtype>> diff = std::make_shared<Tensor<Dtype>>();
		diff->ReshapeLike(*this->top_tensor_);

		FillerParameter filler_param;
		filler_param.set_min(Dtype(0));
		filler_param.set_max(Dtype(1));
		UniformFiller<Dtype> filler(filler_param);
		filler.Fill(diff.get());
		//给输出误差值 初始化随机值0-1
		caffe_copy(this->top_tensor_list_[0]->count(),
			         diff->cpu_data(),
			         this->top_tensor_list_[0]->mutable_cpu_diff());
		vector<bool> propagate_down(1, true);
		layer->Backward(this->top_tensor_list_, propagate_down, this->bottom_tensor_list_);

		//copy 权重data 和梯度diff
		shared_ptr<Tensor<Dtype>> weight = std::make_shared<Tensor<Dtype>>();
		weight->CopyFrom(*layer->weights()[0], false, true);
		weight->CopyFrom(*layer->weights()[0], true, true);

		//copy 输入误差值
		shared_ptr<Tensor<Dtype>> bottom_diff = std::make_shared<Tensor<Dtype>>();
		bottom_diff->CopyFrom(*this->bottom_tensor_list_[0], true, true);

		this->top_tensor_list_.clear();
		this->top_tensor_list_.push_back(new Tensor<Dtype>());
		//权重转置
		this->fully_connected_param_->set_transpose(true);
		shared_ptr<FullyConnectedLayer<Dtype>> layer_transpose(new FullyConnectedLayer<Dtype>(this->layer_param_));
		//层初始化
		layer_transpose->SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
		const Dtype* weight_src = weight->cpu_data(); //10 x 60
		Dtype* weight_transpose = layer_transpose->weights()[0]->mutable_cpu_data(); //60 x 10
		const int width = layer->weights()[0]->shape(1); //60
		const int width_transpose = layer_transpose->weights()[0]->shape(1); //10
		//copy 权重
		for (int i = 0; i < layer->weights()[0]->count(); ++i) {
			int row = i / width; //0-9
			int col = i % width; //0-59
			//把右边每一行 (移动)列的值 赋给左边每一列 (移动)行
			weight_transpose[col * width_transpose + row] = weight_src[row * width + col];
		}
		//copy bias
		ASSERT_EQ(layer->weights()[1]->count(), layer_transpose->weights()[1]->count());
		caffe_copy(layer->weights()[1]->count(), layer->weights()[1]->cpu_data(),
			         layer_transpose->weights()[1]->mutable_cpu_data());

		layer_transpose->Forward(this->bottom_tensor_list_, this->top_tensor_list_);
		caffe_copy(this->top_tensor_list_[0]->count(),
			         diff->cpu_data(),
			         this->top_tensor_list_[0]->mutable_cpu_diff());
		layer_transpose->Backward(this->top_tensor_list_, propagate_down, this->bottom_tensor_list_);
		//不转置 计算的梯度
		const Dtype* gradient = weight->cpu_diff();
		//转置 计算的梯度
		const Dtype* gradient_transpose = layer_transpose->weights()[0]->cpu_diff();
		const int WIDTH = layer->weights()[0]->shape(1);
		const int WIDTH_TRANSPOSE = layer_transpose->weights()[0]->shape(1);
		//用的权重和bias一样 梯度是否相等
		for (int i = 0; i < layer->weights()[0]->count(); ++i) {
			int row = i / WIDTH;
			int col = i % WIDTH;
			EXPECT_NE(Dtype(0), gradient[row * WIDTH + col]);
			EXPECT_FLOAT_EQ(gradient[row * WIDTH + col], gradient_transpose[col * WIDTH_TRANSPOSE + row]);
		}

		//输入的误差值
		gradient = bottom_diff->cpu_diff();
		gradient_transpose = this->bottom_tensor_list_[0]->cpu_diff();
		//用的权重和bias一样 输入误差值是否相等
		for (int i = 0; i < this->bottom_tensor_list_[0]->count(); ++i) {
			EXPECT_NE(Dtype(0), gradient[i]);
			EXPECT_FLOAT_EQ(gradient[i], gradient_transpose[i]);
		}
	} else {
		LOG(ERROR) << "Skipping test due to old architecture";
	}
}


}     //namespace caffe