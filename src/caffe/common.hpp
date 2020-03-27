//
// Created by yipeng on 2020/3/21.
//
#ifndef SIMPLE_CAFFE_COMMON_HPP_
#define SIMPLE_CAFFE_COMMON_HPP_

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <cstdint>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <random>

#include "caffe/util/device_alternate.hpp"

//没有定义gflags 就用google命名空间
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

//禁止类的拷贝构造和拷贝赋值
#define DISABLE_COPY_AND_ASSIGN(classname) \
private: \
  classname(const classname&); \
  classname& operator=(const classname&)

//实例化float和double类型模板类
#define INSTANTIATE_CLASS(classname) \
	template class classname<float>; \
	template class classname<double>

//实例化层的float和double类型 GPU前向计算
#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
	template void classname<float>::Forward_gpu( \
			const std::vector<Tensor<float>*>& bottom, \
			const std::vector<Tensor<float>*>& top); \
	template void classname<double>::Forward_gpu( \
			const std::vector<Tensor<double>*>& bottom, \
			const std::vector<Tensor<double>*>& top);

//实例化层的float和double类型 GPU反向计算
#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
	template void classname<float>::Backward_gpu( \
			const std::vector<Tensor<float>*>& top, \
			const std::vector<bool>& error_propagate_down, \
			const std::vector<Tensor<float>*>& bottom); \
	template void classname<double>::Backward_gpu( \
			const std::vector<Tensor<double>*>& top, \
			const std::vector<bool>& error_propagate_down, \
			const std::vector<Tensor<double>*>& bottom);

//实例化层的float和double类型 GPU前向计算和反向计算
#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
	INSTANTIATE_LAYER_GPU_FORWARD(classname); \
	INSTANTIATE_LAYER_GPU_BACKWARD(classname)

//还没有定义的api返回没有实现
#define NOT_IMPLEMENTED LOG(FATAL) << " That API Not Implemented Yet"

namespace caffe {
using std::ios;
using std::fstream;
using std::stringstream;
using std::string;

using std::vector;
using std::set;
using std::map;
using std::pair;
using std::make_pair;
using std::iterator;

using std::shared_ptr;

void GlobalInit(int* pargc, char*** pargv);

class Caffe {
 public:
	~Caffe();
	static Caffe& Get();

	typedef std::mt19937 rng_t;
	enum Mode { CPU, GPU };

	//内部类 cpu的随机数生成
	class RNG {
	 public:
		RNG();
		explicit RNG(unsigned int seed);
		explicit RNG(const RNG&);
		RNG& operator=(const RNG&);
		void* generator();
	 private:
		class Generator;
		shared_ptr<Generator> generator_;
	};

#ifndef CPU_ONLY
	//得到cublas和curand句柄
	inline static cublasHandle_t cublas_handle() {
		return Get().cublas_handle_;
	}
	inline static curandGenerator_t curand_generator() {
		return Get().curand_generator_;
	}
#endif

	inline static Mode mode() { return Get().mode_; }
	inline static void set_mode(Mode mode) { Get().mode_ = mode; }
	static void set_random_seed(const unsigned int seed);
	static void DeviceQuery();
	static bool CheckDevice(const int device_id);
	static void SetDevice(const int device_id);
	static int FindDevice(const int start_id = 0);

//	new一个RNG对象
	inline static RNG& rng_stream() {
		if (!Get().random_generator_.get()) {
			Get().random_generator_.reset(new RNG());
		}
		return *(Get().random_generator_.get());
	}

 protected:
#ifndef CPU_ONLY
	cublasHandle_t cublas_handle_;
	curandGenerator_t curand_generator_;
#endif
	shared_ptr<RNG> random_generator_;
	Mode mode_;

	//并行训练
	int solver_count_;
	int solver_rank_;
	bool multiprocess_;

 private:
	//私有构造
	Caffe();
	DISABLE_COPY_AND_ASSIGN(Caffe);
};     //class Caffe

}      //namespace caffe

#endif //SIMPLE_CAFFE_COMMON_HPP_
