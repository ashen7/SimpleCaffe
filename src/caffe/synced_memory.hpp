//
// Created by yipeng on 2020/3/22.
//
#ifndef SIMPLE_CAFFE_SYNCED_MEMORY_HPP_
#define SIMPLE_CAFFE_SYNCED_MEMORY_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {
//分配分页锁定内存pinned 比普通的可分页内存 host和device传输拷贝时速度要快
//单GPU提升的性能一般 对并行训练多GPU会提升稳定性
inline void CaffeMallocHost(void **ptr, size_t size, bool *use_cuda) {
#ifndef CPU_ONLY
	if (Caffe::mode() == Caffe::GPU) {
		CUDA_CHECK(cudaMallocHost(ptr, size));
		*use_cuda = true;
		return;
	}
#endif
	*ptr = malloc(size);
	*use_cuda = false;
	CHECK(*ptr) << "host allocation of size" << size << " failed";
}

inline void CaffeFreeHost(void *ptr, bool use_cuda) {
#ifndef CPU_ONLY
	if (use_cuda) {
		CUDA_CHECK(cudaFreeHost(ptr));
		return;
	}
#endif
	free(ptr);
}

//用于同步 cpu 和 gpu之间的数据
class SyncedMemory {
 public:
	enum SyncedHead {
		UNINITIALIZED,  //未初始化
		HEAD_AT_CPU,    //数据在cpu
		HEAD_AT_GPU,    //数据在gpu
		SYNCED          //数据在cpu和gpu同步
	};

	SyncedMemory();
	explicit SyncedMemory(size_t size);
	~SyncedMemory();
	void set_cpu_data(void* data);
	void set_gpu_data(void* data);
	const void* cpu_data();        //返回数据不可改变的cpu_data
	const void* gpu_data();        //返回数据不可改变的gpu_data
	void* mutable_cpu_data();      //返回数据可以改变的cpu_data
	void* mutable_gpu_data();      //返回数据可以改变的gpu_data
	SyncedHead head() const { return head_; }
	size_t size() const { return size_; }

#ifndef CPU_ONLY
	void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
	void check_device();
	void to_cpu();
	void to_gpu();

	void* cpu_data_;
	void* gpu_data_;
	size_t size_;
	SyncedHead head_;
	bool own_cpu_data_;
	bool own_gpu_data_;
	bool cpu_malloc_use_cuda_;
	int device_;

	DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};     //class SyncedMemory

}      //namespace caffe

#endif //SIMPLE_CAFFE_SYNCED_MEMORY_HPP_
