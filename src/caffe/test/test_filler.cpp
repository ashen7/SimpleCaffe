//
// Created by yipeng on 2020/3/24.
//
#include <vector>

#include <gtest/gtest.h>

#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

//ConstantFiller测试固件类模板
template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
	ConstantFillerTest()
			: tensor_(new Tensor<Dtype>()),
			  filler_param_() {
		filler_param_.set_value(10.0);
		filler_.reset(new ConstantFiller<Dtype>(filler_param_));
	}
	virtual void test_params(const vector<int>& shape) {
		EXPECT_TRUE(tensor_);
		tensor_->Reshape(shape);
		filler_->Fill(tensor_);
		const int count = tensor_->count();
		const Dtype* data = tensor_->cpu_data();
		for (int i = 0; i < count; ++i) {
			EXPECT_EQ(data[i], filler_param_.value());
		}
	}
	virtual ~ConstantFillerTest() {
		if (tensor_) {
			delete tensor_;
		}
	}
	
	Tensor<Dtype>* const tensor_;
	FillerParameter filler_param_;
	shared_ptr<ConstantFiller<Dtype>> filler_;
};

//ConstantFiller注册模板参数化
TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
	vector<int> shape{ 2, 3, 4, 5 };
	this->test_params(shape);
}

TYPED_TEST(ConstantFillerTest, TestFill2D) {
	vector<int> shape{ 2, 3 };
	this->test_params(shape);
}

TYPED_TEST(ConstantFillerTest, TestFill5D) {
	vector<int> shape{ 2, 3, 4, 5, 2 };
	this->test_params(shape);
}

//UniformFiller测试固件类模板
template <typename Dtype>
class UniformFillerTest : public ::testing::Test {
 protected:
	UniformFillerTest()
		: tensor_(new Tensor<Dtype>()),
		  filler_param_() {
		filler_param_.set_min(1);
		filler_param_.set_max(2);
		filler_.reset(new UniformFiller<Dtype>(filler_param_));
	}
	virtual void test_params(const vector<int>& shape) {
		EXPECT_TRUE(tensor_);
		tensor_->Reshape(shape);
		filler_->Fill(tensor_);
		const int count = tensor_->count();
		const Dtype* data = tensor_->cpu_data();
		for (int i = 0; i < count; ++i) {
			EXPECT_GE(data[i], filler_param_.min());
			EXPECT_LE(data[i], filler_param_.max());
		}
	}
	virtual ~UniformFillerTest() {
		if (tensor_) {
			delete tensor_;
		}
	}

	Tensor<Dtype>* const tensor_;
	FillerParameter filler_param_;
	shared_ptr<UniformFiller<Dtype>> filler_;
};

//UniformFiller注册模板参数化
TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
	vector<int> shape{ 2, 3, 4, 5 };
	this->test_params(shape);
}

TYPED_TEST(UniformFillerTest, TestFill1D) {
	vector<int> shape(1, 10);
	this->test_params(shape);
}

TYPED_TEST(UniformFillerTest, TestFill2D) {
	vector<int> shape{ 2, 3 };
	this->test_params(shape);
}

TYPED_TEST(UniformFillerTest, TestFill5D) {
	vector<int> shape{ 2, 3, 4, 5, 2 };
	this->test_params(shape);
}

//GaussianFiller测试固件类模板
template <typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
	GaussianFillerTest()
		: tensor_(new Tensor<Dtype>()),
		  filler_param_() {
		filler_param_.set_mean(10.0);
		filler_param_.set_stddev(0.1);
		filler_.reset(new GaussianFiller<Dtype>(filler_param_));
	}

	virtual void test_params(const vector<int>& shape,
													 const Dtype tolerance = Dtype(5),
													 const int repetitions = 100) {
		EXPECT_TRUE(tensor_);
		tensor_->Reshape(shape);
		for (int i = 0; i < repetitions; ++i) {
			test_params_iter(tolerance);
		}
	}

	virtual void test_params_iter(const Dtype tolerance) {
		filler_->Fill(tensor_);
		const int count = tensor_->count();
		const Dtype* data = tensor_->cpu_data();
		Dtype mean = Dtype(0);
		Dtype var = Dtype(0);
		for (int i = 0; i < count; ++i) {
			mean += data[i];
			var += data[i] * data[i];
		}
		mean /= count;
		var /= count;
		var -= mean * mean;
    //均值应该要在 上下标准差的tolerance之内
		EXPECT_GE(mean, filler_param_.mean() - filler_param_.stddev() * tolerance);
		EXPECT_LE(mean, filler_param_.mean() + filler_param_.stddev() * tolerance);

		//方差
		Dtype target_var = filler_param_.stddev() * filler_param_.stddev();
		EXPECT_GE(var, target_var / tolerance);
		EXPECT_LE(var, target_var * tolerance);
	}

	virtual ~GaussianFillerTest() {
		if (tensor_) {
			delete tensor_;
		}
	}

	Tensor<Dtype>* const tensor_;
	FillerParameter filler_param_;
	shared_ptr<GaussianFiller<Dtype>> filler_;
};

//UniformFiller注册模板参数化
TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
	vector<int> shape{ 2, 3, 4, 5 };
	const TypeParam tolerance = TypeParam(3);
	this->test_params(shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill1D) {
	vector<int> shape(1, 125);
	const TypeParam tolerance = TypeParam(3);
	this->test_params(shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill2D) {
	vector<int> shape{ 8, 15 };
	const TypeParam tolerance = TypeParam(3);
	this->test_params(shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill5D) {
	vector<int> shape{ 2, 3, 4, 5, 2 };
	const TypeParam tolerance = TypeParam(2);
	this->test_params(shape, tolerance);
}

//XavierFiller测试固件类模板
template <typename Dtype>
class XavierFillerTest : public ::testing::Test {
protected:
	XavierFillerTest()
		: tensor_(new Tensor<Dtype>()),
		  filler_param_() {
	}

	virtual void test_params(FillerParameter_VarianceNorm variance_norm,
												   Dtype n, const vector<int>& shape,
	                         const int repetitions = 100) {
		EXPECT_TRUE(tensor_);
		tensor_->Reshape(shape);
		for (int i = 0; i < repetitions; ++i) {
			test_params_iter(variance_norm, n);
		}
	}

	virtual void test_params_iter(FillerParameter_VarianceNorm variance_norm, Dtype n) {
		filler_param_.set_variance_norm(variance_norm);
		filler_.reset(new XavierFiller<Dtype>(filler_param_));
		filler_->Fill(tensor_);
		const int count = tensor_->count();
		const Dtype* data = tensor_->cpu_data();
		Dtype mean = Dtype(0);
		Dtype ex2 = Dtype(0);
		for (int i = 0; i < count; ++i) {
			mean += data[i];
			ex2 += data[i] * data[i];
		}
		mean /= count;
		ex2 /= count;
		Dtype stddev = sqrt(ex2 - mean * mean);
		Dtype target_stddev = sqrt(2.0 / n);

		EXPECT_NEAR(mean, 0.0, 0.1);
		EXPECT_NEAR(stddev, target_stddev, 0.1);
	}

	virtual ~XavierFillerTest() {
		if (tensor_) {
			delete tensor_;
		}
	}

	Tensor<Dtype>* const tensor_;
	FillerParameter filler_param_;
	shared_ptr<XavierFiller<Dtype>> filler_;
};

//UniformFiller注册模板参数化
TYPED_TEST_CASE(XavierFillerTest, TestDtypes);

TYPED_TEST(XavierFillerTest, TestFillFanIn) {
	vector<int> shape{ 1000, 2, 4, 5 };
	//fan in n取C*H*W
	TypeParam n = 2 * 4 * 5;
	this->test_params(FillerParameter_VarianceNorm_FAN_IN, n, shape);
}

TYPED_TEST(XavierFillerTest, TestFillFanOut) {
	vector<int> shape{ 1000, 2, 4, 5 };
	//fan in n取N*H*W
	TypeParam n = 1000 * 4 * 5;
	this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n, shape);
}

TYPED_TEST(XavierFillerTest, TestFillAverage) {
	vector<int> shape{ 1000, 2, 4, 5 };
	//fan in n取C*H*W
	TypeParam n = (2 * 4 * 5 + 1000 * 4 * 5) / 2.0;
	this->test_params(FillerParameter_VarianceNorm_AVERAGE, n, shape);
}

TYPED_TEST(XavierFillerTest, TestFill1D) {
	EXPECT_TRUE(this->tensor_);
	vector<int> shape(1, 25);
	this->tensor_->Reshape(shape);
	this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
	this->filler_.reset(new XavierFiller<TypeParam>(this->filler_param_));
	this->filler_->Fill(this->tensor_);
}

TYPED_TEST(XavierFillerTest, TestFill2D) {
	EXPECT_TRUE(this->tensor_);
	vector<int> shape{ 8, 3 };
	this->tensor_->Reshape(shape);
	this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
	this->filler_.reset(new XavierFiller<TypeParam>(this->filler_param_));
	this->filler_->Fill(this->tensor_);
}

TYPED_TEST(XavierFillerTest, TestFill5D) {
	EXPECT_TRUE(this->tensor_);
	vector<int> shape{ 2, 3, 4, 5, 2 };
	this->tensor_->Reshape(shape);
	this->filler_param_.set_variance_norm(FillerParameter_VarianceNorm_AVERAGE);
	this->filler_.reset(new XavierFiller<TypeParam>(this->filler_param_));
	this->filler_->Fill(this->tensor_);
}

}       //namespace caffe