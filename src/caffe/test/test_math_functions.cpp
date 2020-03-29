//
// Created by yipeng on 2020/3/27.
//
#include <stdint.h>
#include <time.h>
#include <cmath>

#include <gtest/gtest.h>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

//测试固件类模板
template <typename TypeParam>
class MathFunctionsTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;

 protected:
	MathFunctionsTest()
			: bottom_tensor_(new Tensor<Dtype>()),
				top_tensor_(new Tensor<Dtype>()) {}

	virtual void SetUp() override {
		Caffe::set_random_seed(1701);
		this->bottom_tensor_->Reshape(11, 17, 19, 23);
		this->top_tensor_->Reshape(11, 17, 19, 23);
		//填充值
		FillerParameter filler_param;
		filler_param.set_mean(Dtype(0));
		filler_param.set_stddev(Dtype(1));
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->bottom_tensor_);
		filler.Fill(this->top_tensor_);
	}

	virtual ~MathFunctionsTest() {
		if (this->bottom_tensor_) {
			delete this->bottom_tensor_;
		}
		if (this->top_tensor_) {
			delete this->top_tensor_;
		}
	}

		Tensor<Dtype>* const bottom_tensor_;
		Tensor<Dtype>* const top_tensor_;
};

template <typename Dtype>
class CPUMathFunctionsTest : public MathFunctionsTest<CPUDevice<Dtype>> {
};

//注册测试用例 模板类型参数化 测试CPU的float double两个类型
TYPED_TEST_CASE(CPUMathFunctionsTest, TestDtypes);

TYPED_TEST(CPUMathFunctionsTest, TestNothing) {
	//测试套件的第一个测试去setup
}

TYPED_TEST(CPUMathFunctionsTest, TestAsum) {
	const int count = this->bottom_tensor_->count();
	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	TypeParam asum = 0;
	for (int i = 0; i < count; ++i) {
		asum += std::fabs(bottom_data[i]);
	}
	TypeParam cpu_asum = caffe_cpu_asum<TypeParam>(count, bottom_data);
	EXPECT_LT((cpu_asum - asum) / asum, 1e-2);
}

TYPED_TEST(CPUMathFunctionsTest, TestSign) {
	const int count = this->bottom_tensor_->count();
	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	caffe_cpu_sign<TypeParam>(count, bottom_data, this->bottom_tensor_->mutable_cpu_diff());

	const TypeParam* signs = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(signs[i], bottom_data[i] > 0 ? 1 : (bottom_data[i] < 0 ? -1 : 0));
	}
}

TYPED_TEST(CPUMathFunctionsTest, TestSgnbit) {
	const int count = this->bottom_tensor_->count();
	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	caffe_cpu_sgnbit<TypeParam>(count, bottom_data, this->bottom_tensor_->mutable_cpu_diff());

	const TypeParam* signbits = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(signbits[i], bottom_data[i] < 0 ? 1 : 0);
	}
}

TYPED_TEST(CPUMathFunctionsTest, TestFabs) {
	const int count = this->bottom_tensor_->count();
	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	caffe_abs<TypeParam>(count, bottom_data, this->bottom_tensor_->mutable_cpu_diff());

	const TypeParam* abs_value = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(abs_value[i], bottom_data[i] >= 0 ? bottom_data[i] : -bottom_data[i]);
	}
}

TYPED_TEST(CPUMathFunctionsTest, TestScale) {
	const int count = this->bottom_tensor_->count();
	TypeParam alpha = this->bottom_tensor_->cpu_diff()[caffe_rng_rand() % count];
	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	caffe_cpu_scale<TypeParam>(count, alpha, bottom_data,
		                         this->bottom_tensor_->mutable_cpu_diff());

	const TypeParam* scale = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(scale[i], bottom_data[i] * alpha);
	}
}

TYPED_TEST(CPUMathFunctionsTest, TestCopy) {
	const int count = this->bottom_tensor_->count();
	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	TypeParam* top_data = this->top_tensor_->mutable_cpu_data();
	caffe_copy(count, bottom_data, top_data);

	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(top_data[i], bottom_data[i]);
	}
}

#ifndef CPU_ONLY

template <typename Dtype>
class GPUMathFunctionsTest : public MathFunctionsTest<GPUDevice<Dtype>> {
};

//注册测试用例 模板类型参数化 测试CPU的float double两个类型
TYPED_TEST_CASE(GPUMathFunctionsTest, TestDtypes);

TYPED_TEST(GPUMathFunctionsTest, TestAsum) {
	const int count = this->bottom_tensor_->count();
	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	TypeParam asum = 0;
	for (int i = 0; i < count; ++i) {
		asum += std::fabs(bottom_data[i]);
	}

	TypeParam gpu_asum;
	caffe_gpu_asum<TypeParam>(count, this->bottom_tensor_->gpu_data(), &gpu_asum);
	EXPECT_LT((gpu_asum - asum) / asum, 1e-2);
}

TYPED_TEST(GPUMathFunctionsTest, TestSign) {
	const int count = this->bottom_tensor_->count();
	caffe_gpu_sign<TypeParam>(count, this->bottom_tensor_->gpu_data(),
		                        this->bottom_tensor_->mutable_gpu_diff());

	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	const TypeParam* signs = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(signs[i], bottom_data[i] > 0 ? 1 : (bottom_data[i] < 0 ? -1 : 0));
	}
}

TYPED_TEST(GPUMathFunctionsTest, TestSgnbit) {
	const int count = this->bottom_tensor_->count();
	caffe_gpu_sgnbit<TypeParam>(count, this->bottom_tensor_->gpu_data(),
		                          this->bottom_tensor_->mutable_gpu_diff());

	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	const TypeParam* signbits = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(signbits[i], bottom_data[i] < 0 ? 1 : 0);
	}
}

TYPED_TEST(GPUMathFunctionsTest, TestFabs) {
	const int count = this->bottom_tensor_->count();
	caffe_gpu_abs<TypeParam>(count, this->bottom_tensor_->gpu_data(),
		                       this->bottom_tensor_->mutable_gpu_diff());

	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	const TypeParam* abs_value = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(abs_value[i], bottom_data[i] >= 0 ? bottom_data[i] : -bottom_data[i]);
	}
}

TYPED_TEST(GPUMathFunctionsTest, TestScale) {
	const int count = this->bottom_tensor_->count();
	TypeParam alpha = this->bottom_tensor_->cpu_diff()[caffe_rng_rand() % count];
	caffe_gpu_scale<TypeParam>(count, alpha, this->bottom_tensor_->gpu_data(),
	                           this->bottom_tensor_->mutable_gpu_diff());

	const TypeParam* bottom_data = this->bottom_tensor_->cpu_data();
	const TypeParam* scale = this->bottom_tensor_->cpu_diff();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(scale[i], bottom_data[i] * alpha);
	}
}

TYPED_TEST(GPUMathFunctionsTest, TestCopy) {
	const int count = this->bottom_tensor_->count();
	const TypeParam* bottom_data = this->bottom_tensor_->gpu_data();
	TypeParam* top_data = this->top_tensor_->mutable_gpu_data();
	caffe_copy(count, bottom_data, top_data);

	bottom_data = this->bottom_tensor_->cpu_data();
	top_data = this->top_tensor_->mutable_cpu_data();
	for (int i = 0; i < count; ++i) {
		EXPECT_EQ(top_data[i], bottom_data[i]);
	}
}


#endif //!CPU_ONLY
}      //namespace caffe