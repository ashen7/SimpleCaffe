//
// Created by yipeng on 2020/3/22.
//
#ifndef CAFFE_TENSOR_HPP_
#define CAFFE_TENSOR_HPP_

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

#include "caffe/common.hpp"
#include "caffe/synced_memory.hpp"
#include "caffe/proto/caffe.pb.h"

const int kMaxTensorDims = 32;

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
  //索引大于0 直接返回对应维度 小于0 返回index + dims()
	inline int CanonicalDimsIndex(int dim_index) const {
		CHECK_GE(dim_index, -num_dims()) << "dim: " << dim_index
			<< " out of range for " << num_dims() << "-D Tensor with shape " << shape_string();
		CHECK_LT(dim_index, num_dims()) << "dim: " << dim_index
		  << " out of range for " << num_dims() << "-D Tensor with shape " << shape_string();
		if (dim_index < 0) {
			return dim_index + num_dims();
		}
		return dim_index;
	}

	inline const vector<int>& shape() const { return shape_; } //得到形状的vector
	inline int num_dims() const { return shape_.size(); } //得到维度 4d 3d 2d 1d
	inline int count() const { return count_; } //得到数据总数

	inline int shape(int index) const {
		return shape_[CanonicalDimsIndex(index)];
	}

	//如果索引在正常范围[-4 - 3] 但是大于了当前shape的维度 就返回1
	inline int LegacyShape(int index) const {
		CHECK_LE(num_dims(), 4) << "tensor dims can not exceed 4";
		CHECK_LT(index, 4);
		CHECK_GE(index, -4);
		if (index >= num_dims() || index < -num_dims()) {
			return 1;
		}
		return shape(index);
	}

	inline int count(int start_dim, int end_dim) const {
		CHECK_LE(start_dim, end_dim);
		CHECK_GE(start_dim, 0);
		CHECK_GE(end_dim, 0);
		CHECK_LE(start_dim, num_dims());
		CHECK_LE(end_dim, num_dims());
		int count = 1;
		for (int i = start_dim; i < end_dim; ++i) {
			count *= shape(i);
		}
		return count;
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

//	通过4个下标来计算偏移量
	inline int offset(const vector<int>& indices) const {
		CHECK_LE(indices.size(), num_dims());
		int offset = 0;
		for (int i = 0; i < num_dims(); ++i) {
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
	shared_ptr<SyncedMemory> data_;  //前向计算的数据 输入/输出
	shared_ptr<SyncedMemory> diff_;  //反向计算的梯度 误差
	vector<int> shape_;              //维度形状 N C H W(batch大小,通道,高,宽)
	int count_;                      //数据总个数
	int capacity_;                   //当前容量

	DISABLE_COPY_AND_ASSIGN(Tensor);
};     //class Tensor

}      //namespace caffe

#endif //CAFFE_TENSOR_HPP_
