//
// Created by yipeng on 2020/3/28.
//
#include <algorithm>
#include <vector>

#include <gtest/gtest.h>
#include <google/protobuf/text_format.h>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/fully_connected_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

//测试固件类模板
template <typename TypeParam>
class ActivationLayerInterfaceTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;
 protected:
	ActivationLayerInterfaceTest()
			: bottom_tensor_(new Tensor<Dtype>(2, 3, 4, 5)),
			  top_tensor_(new Tensor<Dtype>()) {
		Caffe::set_random_seed(1701);
		//填充输入
		FillerParameter filler_param;
		filler_param.set_mean(Dtype(0));
		filler_param.set_stddev(Dtype(1));
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->bottom_tensor_);

		bottom_tensor_list_.push_back(bottom_tensor_);
		top_tensor_list_.push_back(top_tensor_);
	}

	virtual ~ActivationLayerInterfaceTest() {
		if (bottom_tensor_) {
			delete bottom_tensor_;
		}
		if (top_tensor_) {
			delete top_tensor_;
		}
	}

	Tensor<Dtype>* const bottom_tensor_;
	Tensor<Dtype>* const top_tensor_;
	vector<Tensor<Dtype>*> bottom_tensor_list_;
	vector<Tensor<Dtype>*> top_tensor_list_;
};

//注册测试用例 模板类型参数化 测试CPU和GPU的float double两个类型
TYPED_TEST_CASE(ActivationLayerInterfaceTest, TestDtypesAndDevices);

TYPED_TEST(ActivationLayerInterfaceTest, TestReLUForward) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	ReLULayer<Dtype> layer(layer_param);
	//层初始化
	layer.SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	layer.Forward(this->bottom_tensor_list_, this->top_tensor_list_);

	const Dtype* bottom_data = this->bottom_tensor_->cpu_data();
	const Dtype* top_data = this->top_tensor_->cpu_data();
	for (int i = 0; i < this->bottom_tensor_->count(); ++i) {
		EXPECT_GE(top_data[i], 0);
		EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
	}
}

TYPED_TEST(ActivationLayerInterfaceTest, TestReLUGradient) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	ReLULayer<Dtype> layer(layer_param);
	//梯度检查
	GradientChecker<Dtype> gradient_checker(1e-2, 1e-3, 1701, 0, 0.01);
	gradient_checker.CheckGradientElementWise(&layer, this->bottom_tensor_list_, this->top_tensor_list_);
}

TYPED_TEST(ActivationLayerInterfaceTest, TestReLUForwardWithNegativeSlope) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	layer_param.mutable_relu_param()->set_negative_slope(0.01);
	CHECK(google::protobuf::TextFormat::ParseFromString(
			"relu_param { negative_slope: 0.01 }", &layer_param));
	ReLULayer<Dtype> layer(layer_param);
	//层初始化
	layer.SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	layer.Forward(this->bottom_tensor_list_, this->top_tensor_list_);

	const Dtype* bottom_data = this->bottom_tensor_->cpu_data();
	const Dtype* top_data = this->top_tensor_->cpu_data();
	for (int i = 0; i < this->bottom_tensor_->count(); ++i) {
		//>=0就是本身 <0 就是本身*negative_slope
		if (top_data[i] >= 0) {
			EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
		} else {
			EXPECT_FLOAT_EQ(top_data[i], bottom_data[i] * 0.01);
		}
	}
}

TYPED_TEST(ActivationLayerInterfaceTest, TestReLUGradientWithNegativeSlope) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	layer_param.mutable_relu_param()->set_negative_slope(0.01);
	CHECK(google::protobuf::TextFormat::ParseFromString(
		"relu_param { negative_slope: 0.01 }", &layer_param));
	ReLULayer<Dtype> layer(layer_param);
	//梯度检查
	GradientChecker<Dtype> gradient_checker(1e-2, 1e-3, 1701, 0, 0.01);
	gradient_checker.CheckGradientElementWise(&layer, this->bottom_tensor_list_, this->top_tensor_list_);
}

TYPED_TEST(ActivationLayerInterfaceTest, TestSigmoidForward) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	SigmoidLayer<Dtype> layer(layer_param);
	//层初始化
	layer.SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	layer.Forward(this->bottom_tensor_list_, this->top_tensor_list_);

	const Dtype* bottom_data = this->bottom_tensor_->cpu_data();
	const Dtype* top_data = this->top_tensor_->cpu_data();
	for (int i = 0; i < this->bottom_tensor_->count(); ++i) {
		EXPECT_GE(top_data[i], 0);
		EXPECT_LE(top_data[i], 1);
		EXPECT_FLOAT_EQ(top_data[i], 1.0 / (1 + exp(-bottom_data[i])));
	}
}

TYPED_TEST(ActivationLayerInterfaceTest, TestSigmoidGradient) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	SigmoidLayer<Dtype> layer(layer_param);
	//梯度检查
	GradientChecker<Dtype> gradient_checker(1e-2, 1e-3, 1701, 0, 0.01);
	gradient_checker.CheckGradientElementWise(&layer, this->bottom_tensor_list_, this->top_tensor_list_);
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNActivationLayerInterfaceTest : public GPUDeviceTest<Dtype> {
 protected:
	CuDNNActivationLayerInterfaceTest()
			: bottom_tensor_(new Tensor<Dtype>(2, 3, 4, 5)),
			  top_tensor_(new Tensor<Dtype>()) {
		Caffe::set_random_seed(1701);
		//填充输入
		FillerParameter filler_param;
		filler_param.set_mean(Dtype(0));
		filler_param.set_stddev(Dtype(1));
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->bottom_tensor_);

		bottom_tensor_list_.push_back(bottom_tensor_);
		top_tensor_list_.push_back(top_tensor_);
	}

	~CuDNNActivationLayerInterfaceTest() {
		if (bottom_tensor_) {
			delete bottom_tensor_;
		}
		if (top_tensor_) {
			delete top_tensor_;
		}
	}

	Tensor<Dtype>* const bottom_tensor_;
	Tensor<Dtype>* const top_tensor_;
	vector<Tensor<Dtype>*> bottom_tensor_list_;
	vector<Tensor<Dtype>*> top_tensor_list_;
};

TYPED_TEST_CASE(CuDNNActivationLayerInterfaceTest, TestDtypes);

TYPED_TEST(CuDNNActivationLayerInterfaceTest, TestCuDNNReLUForward) {
	LayerParameter layer_param;
	CuDNNReLULayer<TypeParam> layer(layer_param);
	//层初始化
	layer.SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	layer.Forward(this->bottom_tensor_list_, this->top_tensor_list_);

	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	const TypeParam* top_data = this->top_tensor_->cpu_data();
	for (int i = 0; i < this->bottom_tensor_->count(); ++i) {
		EXPECT_GE(top_data[i], 0);
		EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
	}
}

TYPED_TEST(CuDNNActivationLayerInterfaceTest, TestCuDNNReLUGradient) {
	LayerParameter layer_param;
	CuDNNReLULayer<TypeParam> layer(layer_param);
	//梯度检查
	GradientChecker<TypeParam> gradient_checker(1e-2, 1e-3, 1701, 0, 0.01);
	gradient_checker.CheckGradientElementWise(&layer, this->bottom_tensor_list_, this->top_tensor_list_);
}

TYPED_TEST(CuDNNActivationLayerInterfaceTest, TestCuDNNReLUForwardWithNegativeSlope) {
	LayerParameter layer_param;
	layer_param.mutable_relu_param()->set_negative_slope(0.01);
	CHECK(google::protobuf::TextFormat::ParseFromString(
		"relu_param { negative_slope: 0.01 }", &layer_param));
	CuDNNReLULayer<TypeParam> layer(layer_param);
	//层初始化
	layer.SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	layer.Forward(this->bottom_tensor_list_, this->top_tensor_list_);

	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	const TypeParam* top_data = this->top_tensor_->cpu_data();
	for (int i = 0; i < this->bottom_tensor_->count(); ++i) {
		//>=0就是本身 <0 就是本身*negative_slope
		if (top_data[i] >= 0) {
			EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
		} else {
			EXPECT_FLOAT_EQ(top_data[i], bottom_data[i] * 0.01);
		}
	}
}

TYPED_TEST(CuDNNActivationLayerInterfaceTest, TestCuDNNReLUGradientWithNegativeSlope) {
	LayerParameter layer_param;
	layer_param.mutable_relu_param()->set_negative_slope(0.01);
	CHECK(google::protobuf::TextFormat::ParseFromString(
		"relu_param { negative_slope: 0.01 }", &layer_param));
	CuDNNReLULayer<TypeParam> layer(layer_param);
	//梯度检查
	GradientChecker<TypeParam> gradient_checker(1e-2, 1e-3, 1701, 0, 0.01);
	gradient_checker.CheckGradientElementWise(&layer, this->bottom_tensor_list_, this->top_tensor_list_);
}

TYPED_TEST(CuDNNActivationLayerInterfaceTest, TestCuDNNSigmoidForward) {
	LayerParameter layer_param;
	CuDNNSigmoidLayer<TypeParam> layer(layer_param);
	//层初始化
	layer.SetUp(this->bottom_tensor_list_, this->top_tensor_list_);
	layer.Forward(this->bottom_tensor_list_, this->top_tensor_list_);

	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	const TypeParam* top_data = this->top_tensor_->cpu_data();
	for (int i = 0; i < this->bottom_tensor_->count(); ++i) {
		EXPECT_GE(top_data[i], 0);
		EXPECT_LE(top_data[i], 1);
		EXPECT_FLOAT_EQ(top_data[i], 1.0 / (1 + exp(-bottom_data[i])));
	}
}

TYPED_TEST(CuDNNActivationLayerInterfaceTest, TestCuDNNSigmoidGradient) {
	LayerParameter layer_param;
	CuDNNSigmoidLayer<TypeParam> layer(layer_param);
	//梯度检查
	GradientChecker<TypeParam> gradient_checker(1e-2, 1e-3, 1701, 0, 0.01);
	gradient_checker.CheckGradientElementWise(&layer, this->bottom_tensor_list_, this->top_tensor_list_);
}

#endif //USE_CUDNN
}      //namespace caffe