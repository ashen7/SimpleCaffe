//
// Created by yipeng on 2020/3/22.
//
#include <climits>

#include <vector>

#include "caffe/tensor.hpp"
#include "caffe/synced_memory.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
Tensor<Dtype>::Tensor(const int num, const int channels, const int height, const int width)
		: capacity_(0) {
	Reshape(num, channels, height, width);
}

template <typename Dtype>
Tensor<Dtype>::Tensor(const vector<int>& shape)
		: capacity_(0) {
	Reshape(shape);
}

template <typename Dtype>
void Tensor<Dtype>::Reshape(const int num, const int channels, const int height, const int width) {
	vector<int> shape(4);
	shape[0] = num;
	shape[1] = channels;
	shape[2] = height;
	shape[3] = width;
	Reshape(shape);
}

//更新count capacity nwe同步内存对象 重新设置同步内存的大小
//此时不会分配内存 第一次调用数据时才会分配内存
template <typename Dtype>
void Tensor<Dtype>::Reshape(const vector<int>& shape) {
	CHECK_LE(shape.size(), kMaxTensorAxes);
	count_ = 1;
	shape_.resize(shape.size());
	for (int i = 0; i < shape.size(); ++i) {
		CHECK_GE(shape[i], 0);
		if (count_ != 0) {
			CHECK_LE(shape[i], INT_MAX / count_) << "tensor size exceeds INT_MAX";
		}
		count_ *= shape[i];
		shape_[i] = shape[i];
	}
//	先new 一个data和diff对象 到调用cpu_data/gpu_data的时候才会真正分配内存
	if (count_ > capacity_) {
		capacity_ = count_;
		data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
		diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	}
}

template <typename Dtype>
void Tensor<Dtype>::Reshape(const TensorShape& shape) {
	CHECK_LE(shape.dim_size(), kMaxTensorAxes);
	vector<int> shape_list(shape.dim_size());
	for (int i = 0; i < shape.dim_size(); ++i) {
		shape_list[i] = shape.dim(i);
	}
	Reshape(shape_list);
}

template <typename Dtype>
void Tensor<Dtype>::ReshapeLike(const Tensor& other) {
	Reshape(other.shape());
}

template <typename Dtype>
void Tensor<Dtype>::set_cpu_data(Dtype* data) {
	CHECK(data);
	size_t size = count_ * sizeof(Dtype);
	if (data_->size() != size) {
		data_.reset(new SyncedMemory(size));
		diff_.reset(new SyncedMemory(size));
	}
	data_->set_cpu_data(data);
}

template <typename Dtype>
void Tensor<Dtype>::set_gpu_data(Dtype* data) {
	CHECK(data);
	size_t size = count_ * sizeof(Dtype);
	if (data_->size() != size) {
		data_.reset(new SyncedMemory(size));
		diff_.reset(new SyncedMemory(size));
	}
	data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_data() const {
	CHECK(data_);
	return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_diff() const {
	CHECK(diff_);
	return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::gpu_data() const {
	CHECK(data_);
	return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::gpu_diff() const {
	CHECK(diff_);
	return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_cpu_data() {
	CHECK(data_);
	return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_cpu_diff() {
	CHECK(diff_);
	return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_gpu_data() {
	CHECK(data_);
	return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_gpu_diff() {
	CHECK(diff_);
	return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Tensor<Dtype>::SharedData(const Tensor& other) {
	CHECK_EQ(count_, other.count());
	data_ = other.data();
}

template <typename Dtype>
void Tensor<Dtype>::SharedDiff(const Tensor& other) {
	CHECK_EQ(count_, other.count());
	diff_ = other.diff();
}

//更新参数
template <> void Tensor<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Tensor<Dtype>::Update() {
	switch (data_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
			//在cpu执行
			caffe_axpy<Dtype>(count_, Dtype(-1),
												static_cast<const Dtype*>(diff_->cpu_data()),
												static_cast<Dtype*>(data_->mutable_cpu_data()));
			break;
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
			//在gpu执行
			caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
														static_cast<const Dtype*>(diff_->gpu_data()),
														static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
			NO_GPU;
#endif
			break;
		default:
			LOG(FATAL) << "SyncedMemory not initialized";
	}
}

//L1范数 绝对值的和
template <> int Tensor<int>::asum_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::asum_data() const {
	if (!data_) {
		return 0;
	}
	switch (data_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
			return caffe_cpu_asum(count_, cpu_data());
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		{
			//case里因为定义了一个变量 所以要用括号
			Dtype asum;
			caffe_gpu_asum(count_, gpu_data(), &asum);
			return asum;
		}
#else
			NO_GPU;
#endif
		case SyncedMemory::UNINITIALIZED:
			return 0;
		default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	}

	return 0;
}

template <> int Tensor<int>::asum_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}
//L1范数 绝对值的和
template <typename Dtype>
Dtype Tensor<Dtype>::asum_diff() const {
	if (!diff_) {
		return 0;
	}
	switch (diff_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
			return caffe_cpu_asum(count_, cpu_diff());
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		{
			//case里因为定义了一个变量 所以要用括号
			Dtype asum;
			caffe_gpu_asum(count_, gpu_diff(), &asum);
			return asum;
		}
#else
			NO_GPU;
#endif
		case SyncedMemory::UNINITIALIZED:
			return 0;
		default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
	}

	return 0;
}

//L2范数 平方和
template <> int Tensor<int>::sumsq_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::sumsq_data() const {
	Dtype sumsq;
	const Dtype* data;
	if (!data_) {
		return 0;
	}
	switch (data_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
			data = cpu_data();
			sumsq = caffe_cpu_dot(count_, data, data);
			break;
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
			data = gpu_data();
			caffe_gpu_dot(count_, data, data, &sumsq);
#else
			NO_GPU;
#endif
			break;
		case SyncedMemory::UNINITIALIZED:
			return 0;
		default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	}

	return sumsq;
}

//L2范数 平方和
template <> int Tensor<int>::sumsq_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::sumsq_diff() const {
	Dtype sumsq;
	const Dtype* diff;
	if (!diff_) {
		return 0;
	}
	switch (diff_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
			diff = cpu_diff();
			sumsq = caffe_cpu_dot(count_, diff, diff);
			break;
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
			diff = gpu_diff();
			caffe_gpu_dot(count_, diff, diff, &sumsq);
#else
			NO_GPU;
#endif
			break;
		case SyncedMemory::UNINITIALIZED:
			return 0;
		default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
	}

	return sumsq;
}

//缩放张量scale tensor
template <> void Tensor<int>::scale_data(const int scale) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::scale_data(const Dtype scale) {
	Dtype* data;
	if (!data_) {
		return;
	}
	switch (data_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
			data = mutable_cpu_data();
			caffe_scal(count_, scale, data);
			return;
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
			data = mutable_gpu_data();
			caffe_gpu_scal(count_, scale, data);
			return;
#else
			NO_GPU;
#endif
		case SyncedMemory::UNINITIALIZED:
			return;
		default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	}
}

//缩放张量scale tensor
template <> void Tensor<int>::scale_diff(const int scale) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::scale_diff(const Dtype scale) {
	Dtype* diff;
	if (!diff_) {
		return;
	}
	switch (diff_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
			diff = mutable_cpu_diff();
			caffe_scal(count_, scale, diff);
			return;
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
			diff = mutable_gpu_diff();
			caffe_gpu_scal(count_, scale, diff);
			return;
#else
			NO_GPU;
#endif
		case SyncedMemory::UNINITIALIZED:
			return;
		default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
	}
}

//两个tensor是否维度相同
template <typename Dtype>
bool Tensor<Dtype>::ShapeEqual(const TensorProto& other) {
	//如果proto有4个成员中其中一个的值 就用这种方式来判断
	if (other.num() || other.channels() ||
	    other.height() || other.width()) {
		//这里如果是num()默认是LegacyShape(0)开始 这是要用负数索引才能让shape原本没有值的维度为1 有值的填入
		//如果是正数索引 就变成了shape有值的填入了前面num channels 而后面维度为1
		return shape_.size() <= 4 &&
					 other.num() == LegacyShape(-4) &&
					 other.channels() == LegacyShape(-3) &&
					 other.height() == LegacyShape(-2) &&
					 other.width() == LegacyShape(-1);
	}
	//否则就用dim vector 来判断
	vector<int> other_shape(other.shape().dim_size());
	for (int i = 0; i < other.shape().dim_size(); ++i) {
		other_shape[i] = other.shape().dim(i);
	}
	return shape_ == other_shape;
}

template <typename Dtype>
void Tensor<Dtype>::CopyFrom(const Tensor<Dtype>& source, bool copy_diff, bool reshape) {
	if (source.count() != count_ ||
			source.shape() != shape_) {
		if (reshape) {
			ReshapeLike(source);
		} else {
			LOG(FATAL) << "Trying to copy tensor of different sizes";
		}
	}
	switch (Caffe::mode()) {
		case Caffe::GPU:
			if (copy_diff) {
				caffe_copy(count_, source.gpu_diff(),
									 static_cast<Dtype*>(diff_->mutable_gpu_data()));
			} else {
				caffe_copy(count_, source.gpu_data(),
				           static_cast<Dtype*>(data_->mutable_gpu_data()));
			}
			break;
		case Caffe::CPU:
			if (copy_diff) {
				caffe_copy(count_, source.cpu_diff(),
				           static_cast<Dtype*>(diff_->mutable_cpu_data()));
			} else {
				caffe_copy(count_, source.cpu_data(),
				           static_cast<Dtype*>(data_->mutable_cpu_data()));
			}
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode";
	}
}

//to proto 序列化成proto对象保存下来
template <>
void Tensor<float>::ToProto(TensorProto* proto, bool write_diff) const {
	proto->clear_shape();
	for (int i = 0; i < shape_.size(); ++i) {
		//给shape对象添加成员dim
		proto->mutable_shape()->add_dim(shape_[i]);
	}
	proto->clear_data();
	proto->clear_diff();
	const float* data = cpu_data();
	for (int i = 0; i < count_; ++i) {
		proto->add_data(data[i]);
	}

	if (write_diff) {
		const float* diff = cpu_diff();
		for (int i = 0; i < count_; ++i) {
			proto->add_diff(diff[i]);
		}
	}
}

//to proto 序列化成proto对象保存下来
template <>
void Tensor<double>::ToProto(TensorProto* proto, bool write_diff) const {
	proto->clear_shape();
	for (int i = 0; i < shape_.size(); ++i) {
		//给shape对象添加成员dim
		proto->mutable_shape()->add_dim(shape_[i]);
	}
	proto->clear_double_data();
	proto->clear_double_diff();
	const double* data = cpu_data();
	for (int i = 0; i < count_; ++i) {
		proto->add_double_data(data[i]);
	}

	if (write_diff) {
		const double* diff = cpu_diff();
		for (int i = 0; i < count_; ++i) {
			proto->add_double_diff(diff[i]);
		}
	}
}

//from proto反序列化 从proto对象 来得到值
template <typename Dtype>
void Tensor<Dtype>::FromProto(const TensorProto& proto, bool reshape) {
	if (reshape) {
		vector<int> shape;
		shape.resize(proto.shape().dim_size());
		for (int i = 0; i < proto.shape().dim_size(); ++i) {
			shape[i] = proto.shape().dim(i);
		}
		Reshape(shape);
	} else {
		CHECK(ShapeEqual(proto)) << "shape mismatch (reshape not set)";
	}

	//copy data
	Dtype* data = mutable_cpu_data();
	if (proto.double_data_size() > 0) {
		CHECK_EQ(count_, proto.double_data_size());
		for (int i = 0; i < count_; ++i) {
			data[i] = proto.double_data(i);
		}
	} else {
		CHECK_EQ(count_, proto.data_size());
		for (int i = 0; i < count_; ++i) {
			data[i] = proto.data(i);
		}
	}

	//copy diff
	if (proto.double_diff_size() > 0) {
		CHECK_EQ(count_, proto.double_diff_size());
		Dtype* diff = mutable_cpu_diff();
		for (int i = 0; i < count_; ++i) {
			diff[i] = proto.double_diff(i);
		}
	} else if (proto.diff_size() > 0) {
		CHECK_EQ(count_, proto.diff_size());
		Dtype* diff = mutable_cpu_diff();
		for (int i = 0; i < count_; ++i) {
			diff[i] = proto.diff(i);
		}
	}
}

//注册float double的tensor类
INSTANTIATE_CLASS(Tensor);
template class Tensor<int>;
}       //namespace caffe

//int main() {
//	std::vector<int> shape{2,3};
//	caffe::Tensor<float> tensor(shape);
//	LOG(INFO) << tensor.LegacyShape(-4);
//	LOG(INFO) << tensor.LegacyShape(-3);
//	LOG(INFO) << tensor.LegacyShape(-2);
//	LOG(INFO) << tensor.LegacyShape(-1);
//
//	LOG(INFO) << tensor.num();
//	LOG(INFO) << tensor.channels();
//	LOG(INFO) << tensor.height();
//	LOG(INFO) << tensor.width();
//}