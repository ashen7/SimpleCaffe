//
// Created by yipeng on 2020/3/22.
//
#ifndef SIMPLE_CAFFE_TENSOR_HPP_
#define SIMPLE_CAFFE_TENSOR_HPP_

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

#include "caffe/common.hpp"
#include "caffe/synced_memory.hpp"
#include "caffe/proto/caffe.pb.h"

const int kMaxTensorAxes = 32;

namespace caffe {

//tensor 张量 封装数据传输的媒介
template <typename Dtype>
class Tensor {
 public:
	Tensor()
			: data_(), diff_(), count_(0), capacity_(0) {}

	//通过NCHW来初始化
	explicit Tensor(const int num, const int channels, const int height, const int width);
	//通过vector 4个值 分别代表NCHW来初始化
	explicit Tensor(const vector<int>& shape);

	//都是转换成vector shape去调用的reshape版本
	void Reshape(const int num, const int channels, const int height, const int width);
	void Reshape(const vector<int>& shape);
	void Reshape(const TensorShape& shape);
	void ReshapeLike(const Tensor& other);

//	打印形状和数据总数
	inline string shape_string() const {
		stringstream shape_str;
		for (int i = 0; i < shape_.size(); ++i) {
			shape_str << shape_[i] << " ";
		}
		shape_str << "(" << count_ << ")";
		return shape_str.str();
	}

	//规范的索引 通过正负索引都可返回维度  可以传-1得到最后一个维度
  //索引大于0 直接返回对应维度 小于0 返回index + axes()
	inline int CanonicalAxisIndex(int axis_index) const {
		CHECK_GE(axis_index, -num_axes()) << "axis: " << axis_index
			<< " out of range for " << num_axes() << "-D Tensor with shape " << shape_string();
		CHECK_LT(axis_index, num_axes()) << "axis: " << axis_index
		  << " out of range for " << num_axes() << "-D Tensor with shape " << shape_string();
		if (axis_index < 0) {
			return axis_index + num_axes();
		}
		return axis_index;
	}

	inline const vector<int>& shape() const { return shape_; } //得到形状的vector
	inline int num_axes() const { return shape_.size(); }      //得到维度/轴个数/秩
	inline int count() const { return count_; }                //得到数据大小

	inline int shape(int index) const {
		return shape_[CanonicalAxisIndex(index)];
	}

	//如果索引在正常范围[-4 - 3] 但是大于了当前shape的维度 就返回1
	inline int LegacyShape(int index) const {
		CHECK_LE(num_axes(), 4) << "tensor axes can not exceed 4";
		CHECK_LT(index, 4);
		CHECK_GE(index, -4);
		if (index >= num_axes() || index < -num_axes()) {
			return 1;
		}
		return shape(index);
	}

	//根据开始轴和结束轴得到数据数量
	inline int count(int start_axis, int end_axis) const {
		CHECK_LE(start_axis, end_axis);
		CHECK_GE(start_axis, 0);
		CHECK_GE(end_axis, 0);
		CHECK_LE(start_axis, num_axes());
		CHECK_LE(end_axis, num_axes());
		int count = 1;
		for (int i = start_axis; i < end_axis; ++i) {
			count *= shape(i);
		}
		return count;
	}

	//从开始轴到最后一个轴得到数据数量
	inline int count(int start_axis) const {
		return count(start_axis, num_axes());
	}

	//得到N C H W 如果合理范围但是越界 维度返回1
	inline int num() const { return LegacyShape(0); }
	inline int channels() const { return LegacyShape(1); }
	inline int height() const { return LegacyShape(2); }
	inline int width() const { return LegacyShape(3); }

	//通过n c h w当前下标 计算出一维的偏移量
	inline int offset(const int n, const int c, const int h, const int w) const {
		CHECK_GE(n, 0);
		CHECK_LE(n, num());
		CHECK_GE(channels(), 0);
		CHECK_LE(c, channels());
		CHECK_GE(height(), 0);
		CHECK_LE(h, height());
		CHECK_GE(width(), 0);
		CHECK_LE(w, width());
//		return ((n * channels() + c) * height() + h) * width() + w;
		return n * channels() * height() * width() +
					 c * height() * width() +
					 h * width() + w;
	}

	//通过4个轴下标来计算偏移量
	inline int offset(const vector<int>& indices) const {
		CHECK_LE(indices.size(), num_axes());
		int offset = 0;
		for (int i = 0; i < num_axes(); ++i) {
			offset *= shape(i);
			if (indices.size() > i) {
				CHECK_GE(indices[i], 0);
				CHECK_LE(indices[i], shape(i));
				offset += indices[i];
			}
		}
		return offset;
	}

	//通过N C H W得到偏移量 返回对应下标值
	inline Dtype data_at(const int n, const int c, const int h, const int w) const {
		return cpu_data()[offset(n, c, h, w)];
	}
	inline Dtype diff_at(const int n, const int c, const int h, const int w) const {
		return cpu_diff()[offset(n, c, h, w)];
	}
	inline Dtype data_at(const vector<int>& indices) const {
		return cpu_data()[offset(indices)];
	}
	inline Dtype diff_at(const vector<int>& indices) const {
		return cpu_diff()[offset(indices)];
	}

	//得到数据
	inline const shared_ptr<SyncedMemory>& data() const {
		CHECK(data_);
		return data_;
	}
	inline const shared_ptr<SyncedMemory>& diff() const {
		CHECK(diff_);
		return diff_;
	}

	//得到数据和梯度
	void set_cpu_data(Dtype* data);
	void set_gpu_data(Dtype* data);
	const Dtype* cpu_data() const;
	const Dtype* cpu_diff() const;
	const Dtype* gpu_data() const;
	const Dtype* gpu_diff() const;
	Dtype* mutable_cpu_data();
	Dtype* mutable_cpu_diff();
	Dtype* mutable_gpu_data();
	Dtype* mutable_gpu_diff();
	void Update();

	//从另一个tensor copy过来
	void CopyFrom(const Tensor<Dtype>& source, bool copy_diff = false, bool reshape = false);

	void ToProto(TensorProto* proto, bool write_diff = false) const; //proto序列化
	void FromProto(const TensorProto& proto, bool reshape = true);   //proto反序列化

  //L1范数 绝对值的和
	Dtype asum_data() const;
	Dtype asum_diff() const;
	//L2范数 平方和
	Dtype sumsq_data() const;
	Dtype sumsq_diff() const;

	//缩放张量scale tensor
	void scale_data(const Dtype scale);
	void scale_diff(const Dtype scale);

	//共享数据 两个Tensor共享一个数据指针
	void SharedData(const Tensor& other);
	void SharedDiff(const Tensor& other);

	//判断 两个tensor是否形状一样
	bool ShapeEqual(const TensorProto& other);

 protected:
	shared_ptr<SyncedMemory> data_;  //作为bottom/top时 data存放的是输入输出值    作为weights时 data存放的是权重
	shared_ptr<SyncedMemory> diff_;  //作为bottom/top时 diff存放的是输入输出误差值 作为weights时 diff存放的是梯度
	vector<int> shape_;              //维度形状 N C H W(batch_size,通道,高,宽)
	int count_;                      //数据总个数
	int capacity_;                   //容量

	DISABLE_COPY_AND_ASSIGN(Tensor);
};     //class Tensor

}      //namespace caffe

#endif //SIMPLE_CAFFE_TENSOR_HPP_
