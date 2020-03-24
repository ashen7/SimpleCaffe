//
// Created by yipeng on 2020/3/24.
//
#include <vector>

#include <gtest/gtest.h>

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

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




}       //namespace caffe