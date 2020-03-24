//
// Created by yipeng on 2020/3/23.
//
#include <vector>

#include <gtest/gtest.h>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

//测试固件类模板
template <typename Dtype>
class TensorTest : public ::testing::Test {
 protected:
	TensorTest()
			: tensor_(new Tensor<Dtype>()),
			  tensor_preshaped_(new Tensor<Dtype>(2, 3, 4, 5)) {}
	virtual ~TensorTest() override {
		if (tensor_) {
			delete tensor_;
		}
		if (tensor_preshaped_) {
			delete tensor_preshaped_;
		}
	}

	Tensor<Dtype>* const tensor_;
	Tensor<Dtype>* const tensor_preshaped_;
};

//注册测试用例 模板类型参数化 测试float double两个类型
TYPED_TEST_CASE(TensorTest, TestDtypes);

//模板类型参数化的 test测试宏
TYPED_TEST(TensorTest, TestInit) {
	EXPECT_TRUE(this->tensor_);
	EXPECT_TRUE(this->tensor_preshaped_);
	EXPECT_EQ(this->tensor_preshaped_->num(), 2);
	EXPECT_EQ(this->tensor_preshaped_->channels(), 3);
	EXPECT_EQ(this->tensor_preshaped_->height(), 4);
	EXPECT_EQ(this->tensor_preshaped_->width(), 5);
	EXPECT_EQ(this->tensor_preshaped_->count(), 120);
	EXPECT_EQ(this->tensor_preshaped_->num_dims(), 4);
	EXPECT_EQ(this->tensor_->count(), 0);
	EXPECT_EQ(this->tensor_->num_dims(), 0);
}

TYPED_TEST(TensorTest, TestCPUGPUData) {
	EXPECT_TRUE(this->tensor_preshaped_->cpu_data());
	EXPECT_TRUE(this->tensor_preshaped_->gpu_data());
	EXPECT_TRUE(this->tensor_preshaped_->mutable_cpu_data());
	EXPECT_TRUE(this->tensor_preshaped_->mutable_gpu_data());
}

TYPED_TEST(TensorTest, TestReshape) {
	this->tensor_->Reshape(2, 3, 4, 5);
	EXPECT_EQ(this->tensor_->num(), 2);
	EXPECT_EQ(this->tensor_->channels(), 3);
	EXPECT_EQ(this->tensor_->height(), 4);
	EXPECT_EQ(this->tensor_->width(), 5);
	EXPECT_EQ(this->tensor_->count(), 120);
	EXPECT_EQ(this->tensor_->num_dims(), 4);
}

TYPED_TEST(TensorTest, TestReshapeZero) {
	vector<int> shape(2);
	shape[0] = 0;
	shape[1] = 5;
	this->tensor_->Reshape(shape);
	EXPECT_EQ(this->tensor_->count(), 0);
}

TYPED_TEST(TensorTest, TestTensorProtoShapeEqual) {
	TensorProto tensor_proto;
	vector<int> shape(2);
	shape[0] = 3;
	shape[1] = 2;
	this->tensor_->Reshape(shape);

	//3 x 2 tensor == 1 x 1 x 3 x 2 tensor
	tensor_proto.set_num(1);
	tensor_proto.set_channels(1);
	tensor_proto.set_height(3);
	tensor_proto.set_width(2);
	EXPECT_TRUE(this->tensor_->ShapeEqual(tensor_proto));

	//3 x 2 tensor != 0 x 1 x 3 x 2 tensor
	tensor_proto.set_num(0);
	EXPECT_FALSE(this->tensor_->ShapeEqual(tensor_proto));

	//3 x 2 tensor != 3 x 1 x 3 x 2 tensor
	tensor_proto.set_num(3);
	EXPECT_FALSE(this->tensor_->ShapeEqual(tensor_proto));

	shape.insert(shape.begin(), 1);
	this->tensor_->Reshape(shape);

	//1 x 3 x 2 tensor == 1 x 1 x 3 x 2 tensor
	tensor_proto.set_num(1);
	EXPECT_TRUE(this->tensor_->ShapeEqual(tensor_proto));

	shape[0] = 2;
	this->tensor_->Reshape(shape);

	//2 x 3 x 2 tensor != 1 x 1 x 3 x 2 tensor
	EXPECT_FALSE(this->tensor_->ShapeEqual(tensor_proto));
}

//测试固件类模板
template <typename TypeParam>
class TensorMathTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;
 protected:
	TensorMathTest()
			: tensor_(new Tensor<Dtype>(2, 3, 4, 5)),
			  epsilon_(1e-6) {}
	virtual ~TensorMathTest() {
		if (tensor_) {
			delete tensor_;
		}
	}

	Tensor<Dtype>* const tensor_;
	Dtype epsilon_;
};

//注册测试用例 模板类型参数化 测试CPU和GPU的float double两个类型
TYPED_TEST_CASE(TensorMathTest, TestDtypesAndDevices);

TYPED_TEST(TensorMathTest, TestSumOfSquares) {
	typedef typename TypeParam::Dtype Dtype;

	//未初始化的tensor 平方和为0
	EXPECT_EQ(0, this->tensor_->sumsq_data());
	EXPECT_EQ(0, this->tensor_->sumsq_diff());

  //均匀分布
	FillerParameter filler_param;
	filler_param.set_min(-3);
	filler_param.set_max(3);
	UniformFiller<Dtype> filler(filler_param);
	filler.Fill(this->tensor_);

	//检查data的sumsq
	Dtype expected_sumsq = 0;
	const Dtype* data = this->tensor_->cpu_data();
	for (int i = 0; i< this->tensor_->count(); ++i) {
		expected_sumsq += data[i] * data[i];
	}
	//两个线程 分别得到cpu 和 gpu的data
	switch(TypeParam::device) {
	case Caffe::CPU:
		this->tensor_->mutable_cpu_data();
		break;
	case Caffe::GPU:
		this->tensor_->mutable_gpu_data();
		break;
	default:
		LOG(FATAL) << "Unknown device: " << TypeParam::device;
	}

	//浮点数近似 第三个参数是阈值
	EXPECT_NEAR(expected_sumsq, this->tensor_->sumsq_data(),
							this->epsilon_ * expected_sumsq);
	EXPECT_EQ(0, this->tensor_->sumsq_diff());

	//检查diff的sumsq
	const Dtype kDiffScale = 7;
	caffe_cpu_scale(this->tensor_->count(), kDiffScale, data,
		              this->tensor_->mutable_cpu_diff());
	//两个线程 分别得到cpu 和 gpu的diff
	switch(TypeParam::device) {
		case Caffe::CPU:
			this->tensor_->mutable_cpu_diff();
			break;
		case Caffe::GPU:
			this->tensor_->mutable_gpu_diff();
			break;
		default:
			LOG(FATAL) << "Unknown device: " << TypeParam::device;
	}

	EXPECT_NEAR(expected_sumsq, this->tensor_->sumsq_data(),
	            this->epsilon_ * expected_sumsq);
	const Dtype expected_sumsq_diff =
			expected_sumsq * kDiffScale * kDiffScale;
	EXPECT_NEAR(expected_sumsq_diff, this->tensor_->sumsq_diff(),
		         this->epsilon_ * expected_sumsq_diff);
}

TYPED_TEST(TensorMathTest, TestAsum) {
	typedef typename TypeParam::Dtype Dtype;

	//未初始化的tensor 绝对值的和为0
	EXPECT_EQ(0, this->tensor_->asum_data());
	EXPECT_EQ(0, this->tensor_->asum_diff());

	//均匀分布
	FillerParameter filler_param;
	filler_param.set_min(-3);
	filler_param.set_max(3);
	UniformFiller<Dtype> filler(filler_param);
	filler.Fill(this->tensor_);

	//检查data的asum
	Dtype expected_asum = 0;
	const Dtype* data = this->tensor_->cpu_data();
	for (int i = 0; i< this->tensor_->count(); ++i) {
		expected_asum += std::fabs(data[i]);
	}
	//两个线程 分别得到cpu 和 gpu的data
	switch(TypeParam::device) {
		case Caffe::CPU:
			this->tensor_->mutable_cpu_data();
			break;
		case Caffe::GPU:
			this->tensor_->mutable_gpu_data();
			break;
		default:
			LOG(FATAL) << "Unknown device: " << TypeParam::device;
	}

	//浮点数近似 第三个参数是阈值
	EXPECT_NEAR(expected_asum, this->tensor_->asum_data(),
	            this->epsilon_ * expected_asum);
	EXPECT_EQ(0, this->tensor_->asum_diff());

	//检查diff的asum
	const Dtype kDiffScale = 7;
	caffe_cpu_scale(this->tensor_->count(), kDiffScale, data,
	                this->tensor_->mutable_cpu_diff());
	//两个线程 分别得到cpu 和 gpu的diff
	switch(TypeParam::device) {
		case Caffe::CPU:
			this->tensor_->mutable_cpu_diff();
			break;
		case Caffe::GPU:
			this->tensor_->mutable_gpu_diff();
			break;
		default:
			LOG(FATAL) << "Unknown device: " << TypeParam::device;
	}

	EXPECT_NEAR(expected_asum, this->tensor_->asum_data(),
	            this->epsilon_ * expected_asum);
	const Dtype expected_asum_diff = expected_asum * kDiffScale;
	EXPECT_NEAR(expected_asum_diff, this->tensor_->asum_diff(),
	            this->epsilon_ * expected_asum_diff);
}

TYPED_TEST(TensorMathTest, TestScaleData) {
	typedef typename TypeParam::Dtype Dtype;

	EXPECT_EQ(0, this->tensor_->asum_data());
	EXPECT_EQ(0, this->tensor_->asum_diff());
	FillerParameter filler_param;
	filler_param.set_min(-3);
	filler_param.set_max(3);
	UniformFiller<Dtype> filler(filler_param);
	filler.Fill(this->tensor_);
	const Dtype asum_before_scale = this->tensor_->asum_data();

	//两个线程 分别得到cpu 和 gpu的data
	switch(TypeParam::device) {
		case Caffe::CPU:
			this->tensor_->mutable_cpu_data();
			break;
		case Caffe::GPU:
			this->tensor_->mutable_gpu_data();
			break;
		default:
			LOG(FATAL) << "Unknown device: " << TypeParam::device;
	}

	const Dtype kDataScale = 3;
	this->tensor_->scale_data(3);
	EXPECT_NEAR(asum_before_scale * kDataScale, this->tensor_->asum_data(),
		          this->epsilon_ * asum_before_scale * kDataScale);
	EXPECT_EQ(0, this->tensor_->asum_diff());

	const Dtype kDataToDiffScale = 7;
	const Dtype* data = this->tensor_->cpu_data();
	caffe_cpu_scale(this->tensor_->count(), kDataToDiffScale, data,
		              this->tensor_->mutable_cpu_diff());
	const Dtype expected_asum_before_sacle = asum_before_scale * kDataScale;
	EXPECT_NEAR(expected_asum_before_sacle, this->tensor_->asum_data(),
		          this->epsilon_ * expected_asum_before_sacle);

	const Dtype expected_asum_before_sacle_diff =
			asum_before_scale * kDataScale * kDataToDiffScale;
	EXPECT_NEAR(expected_asum_before_sacle_diff, this->tensor_->asum_diff(),
		          this->epsilon_ * expected_asum_before_sacle_diff);
	//两个线程 分别得到cpu 和 gpu的diff
	switch(TypeParam::device) {
		case Caffe::CPU:
			this->tensor_->mutable_cpu_diff();
			break;
		case Caffe::GPU:
			this->tensor_->mutable_gpu_diff();
			break;
		default:
			LOG(FATAL) << "Unknown device: " << TypeParam::device;
	}

	const Dtype kDiffScale = 3;
	this->tensor_->scale_diff(kDiffScale);
	EXPECT_NEAR(asum_before_scale * kDataScale, this->tensor_->asum_data(),
	            this->epsilon_ * asum_before_scale * kDataScale);
	const Dtype expected_diff_asum =
			expected_asum_before_sacle_diff * kDiffScale;
	EXPECT_NEAR(expected_diff_asum, this->tensor_->asum_diff(),
		          this->epsilon_ * expected_diff_asum);
}

}        //namespace caffe