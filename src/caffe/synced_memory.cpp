//
// Created by yipeng on 2020/3/22.
//
#include "caffe/synced_memory.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
SyncedMemory::SyncedMemory()
		: cpu_data_(nullptr), gpu_data_(nullptr), size_(0),
			head_(UNINITIALIZED), own_cpu_data_(false),
			own_gpu_data_(false), cpu_malloc_use_cuda_(false) {}

SyncedMemory::SyncedMemory(size_t size)
	: cpu_data_(nullptr), gpu_data_(nullptr), size_(size),
	  head_(UNINITIALIZED), own_cpu_data_(false),
	  own_gpu_data_(false), cpu_malloc_use_cuda_(false) {}

SyncedMemory::~SyncedMemory() {
	check_device();
	if (cpu_data_ && own_cpu_data_) {
		CaffeFreeHost(cpu_data_, cpu_malloc_use_cuda_);
	}

#ifndef CPU_ONLY
	if (gpu_data_ && own_gpu_data_) {
		CUDA_CHECK(cudaFree(gpu_data_));
	}
#endif
}

inline void SyncedMemory::to_cpu() {
	check_device();

	switch (head_) {
	case UNINITIALIZED:
		CaffeMallocHost(&cpu_data_, size_, &cpu_malloc_use_cuda_);
		caffe_memset(size_, 0, cpu_data_);
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
		break;
	case HEAD_AT_GPU:
#ifndef CPU_ONLY
		if (nullptr == cpu_data_) {
			CaffeMallocHost(&cpu_data_, size_, &cpu_malloc_use_cuda_);
			own_cpu_data_ = true;
		}
		//数据在gpu 现在同步to cpu 把数据从device端拷到host端来
		caffe_gpu_memcpy(size_, gpu_data_, cpu_data_);
		head_ = SYNCED;
#else
		NO_GPU;
#endif
		break;
	case HEAD_AT_CPU:
		break;
	case SYNCED:
		break;
	}
}

inline void SyncedMemory::to_gpu() {
	check_device();

#ifndef CPU_ONLY
	switch (head_) {
	case UNINITIALIZED:
		CUDA_CHECK(cudaMalloc(&gpu_data_, size_));
		caffe_gpu_memset(size_, 0, gpu_data_);
		head_ = HEAD_AT_GPU;
		own_gpu_data_ = true;
		break;
	case HEAD_AT_CPU:
		if (nullptr == gpu_data_) {
			CUDA_CHECK(cudaMalloc(&gpu_data_, size_));
			own_gpu_data_ = true;
		}
		//数据在cpu 现在同步to gpu 把数据从host端拷到device端来
		caffe_gpu_memcpy(size_, cpu_data_, gpu_data_);
		head_ = SYNCED;
		break;
	case HEAD_AT_GPU:
		break;
	case SYNCED:
		break;
	}
#else
	NO_GPU;
#endif
}

void SyncedMemory::set_cpu_data(void* data) {
	check_device();
	CHECK(data);
	if (own_cpu_data_) {
		CaffeFreeHost(cpu_data_, cpu_malloc_use_cuda_);
	}
	cpu_data_ = data;
	head_ = HEAD_AT_CPU;
	own_cpu_data_ = false;
}

void SyncedMemory::set_gpu_data(void* data) {
	check_device();
#ifndef CPU_ONLY
	CHECK(data);
	if (own_gpu_data_) {
		CUDA_CHECK(cudaFree(gpu_data_));
	}
	gpu_data_ = data;
	head_ = HEAD_AT_GPU;
	own_gpu_data_ = false;
#else
	NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
	check_device();
	to_cpu();
	return (const void*)cpu_data_;
}

const void* SyncedMemory::gpu_data() {
	check_device();
#ifndef CPU_ONLY
	to_gpu();
	return (const void*)gpu_data_;
#else
	NO_GPU;
	return nullptr;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
	check_device();
	to_cpu();
	head_ = HEAD_AT_CPU;
	return cpu_data_;
}

void* SyncedMemory::mutable_gpu_data() {
	check_device();
#ifndef CPU_ONLY
	to_gpu();
	head_ = HEAD_AT_GPU;
	return gpu_data_;
#else
	NO_GPU;
	return nullptr;
#endif
}

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
	int device;
	cudaGetDevice(&device);
	CHECK(device == device_);
	if (gpu_data_ && own_gpu_data_) {
		cudaPointerAttributes attributes;
		CUDA_CHECK(cudaPOinterGetAttributes(&attributes, gpu_data_));
		CHECK(attributes.device == device_);
	}
#endif
#endif
}

}       //namespace caffe