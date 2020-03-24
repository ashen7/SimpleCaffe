//
// Created by yipeng on 2020/3/21.
//
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <cmath>
#include <ctime>
#include <cstdio>

#include <thread>
#include <random>

#include "caffe/common.hpp"

namespace caffe {
//静态全局变量 Caffe对象的智能指针 并且每个线程都有自己的局部数据
static thread_local shared_ptr<Caffe> thread_instance_;

//每个线程调用Get 先判断是否初始化过 没有则new一个对象
Caffe& Caffe::Get() {
	if (!thread_instance_.get()) {
		thread_instance_.reset(new Caffe());
	}
	return *(thread_instance_.get());
}

//全局初始化
void GlobalInit(int* pargc, char*** pargv) {
	//Google gflags
	gflags::ParseCommandLineFlags(pargc, pargv, true);
	//Google glog
	google::InitGoogleLogging(*(pargv)[0]);
	//提供一个fatal退出程序时的处理函数
	google::InstallFailureSignalHandler();
}

//生成随机数种子
int64_t generate_seed() {
	int64_t s, seed, pid;
	FILE* f = fopen("/dev/urandom", "rb");
	if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
		fclose(f);
		return seed;
	}

	LOG(INFO) << "System entropy source not available, "
	             "using fallback algorithm to generate seed instead.";
	if (f)
		fclose(f);

	pid = getpid();
	s = time(nullptr);
	seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
	return seed;
}

//通过设置随机数种子 初始化随机数生成引擎
class Caffe::RNG::Generator {
public:
	Generator()
		: rng_(new Caffe::rng_t(generate_seed())) {}
	explicit Generator(unsigned int seed)
		: rng_(new std::mt19937(seed)) {}
	Caffe::rng_t* rng() { return rng_.get(); }
private:
	shared_ptr<Caffe::rng_t> rng_;
};

Caffe::RNG::RNG()
	: generator_(new Generator()) {}

Caffe::RNG::RNG(unsigned int seed)
	: generator_(new Generator(seed)) {}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
	generator_ = other.generator_;
	return *this;
}

void* Caffe::RNG::generator() {
	return static_cast<void*>(generator_->rng());
}

#ifdef CPU_ONLY //CPU Mode
Caffe::Caffe()
		: random_generator_(), mode_(Caffe::CPU), solver_count_(1),
			solver_rank_(0), multiprocess_(false) {}

Caffe::~Caffe() {}

void Caffe::set_random_seed(const unsigned int seed) {
	Get().random_generator_.reset(new RNG(seed));
}

void Caffe::DeviceQuery() {
	NO_GPU;
}

void Caffe::SetDevice(const int device_id) {
	NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
	NO_GPU;
	return false;
}

int Caffe::FindDevice(const int start_id) {
	NO_GPU;
	return -1;
}

#else  //GPU + CPU Mode
Caffe::Caffe()
		: cublas_handle_(nullptr), curand_generator_(nullptr),
			random_generator_(), mode_(Caffe::GPU),
		  solver_count_(1), solver_rank_(0), multiprocess_(false) {
	if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
		LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available";
	}
	if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
			!= CURAND_STATUS_SUCCESS ||
			curandSetPseudoRandomGeneratorSeed(curand_generator_, generate_seed())
			!= CURAND_STATUS_SUCCESS) {
		LOG(ERROR) << "Cannot create Curand generator. Curand won't be available";
	}
}

Caffe::~Caffe() {
	if (cublas_handle_) {
		CUBLAS_CHECK(cublasDestroy(cublas_handle_));
	}
	if (curand_generator_) {
		CURAND_CHECK(curandDestroyGenerator(curand_generator_));
	}
}

void Caffe::set_random_seed(const unsigned int seed) {
	static bool curand_availability = false;
	if (Get().curand_generator_) {
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(), seed));
		CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
	} else {
		if (!curand_availability) {
			LOG(ERROR) << "Curand not available. Skipping setting the curand seed";
			curand_availability = true;
		}
	}
	Get().random_generator_.reset(new RNG(seed));
}

void Caffe::DeviceQuery() {
	cudaDeviceProp prop;
	int device;
	if (cudaSuccess != cudaGetDevice(&device)) {
		printf("No cuda device present.\n");
		return;
	}
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	std::cout << "====================================================================================================" << std::endl;
	LOG(INFO) << "Device id:                     " << device;
	LOG(INFO) << "Major revision number:         " << prop.major;
	LOG(INFO) << "Minor revision number:         " << prop.minor;
	LOG(INFO) << "Name:                          " << prop.name;
	LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem / 1024.0 / 1024.0 << "MB";
	LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << "KB";
	LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
	LOG(INFO) << "Warp size:                     " << prop.warpSize;
	LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
	LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
	LOG(INFO) << "Maximum dimension of block:    "
	          << prop.maxThreadsDim[0] <<     ", " << prop.maxThreadsDim[1] << ", "
	          << prop.maxThreadsDim[2];
	LOG(INFO) << "Maximum dimension of grid:     "
	          << prop.maxGridSize[0] <<       ", " << prop.maxGridSize[1] << ", "
	          << prop.maxGridSize[2];
	LOG(INFO) << "Clock rate:                    " << prop.clockRate;
	LOG(INFO) << "Total constant memory:         " << prop.totalConstMem / 1024.0 << "KB";
	LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
	LOG(INFO) << "Concurrent copy and execution: "
	          << (prop.deviceOverlap ? "Yes" : "No");
	LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
	LOG(INFO) << "Kernel execution timeout:      "
	          << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
	std::cout << "====================================================================================================" << std::endl;
}

void Caffe::SetDevice(const int device_id) {
	int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return ;
  }
  CUDA_CHECK(cudaSetDevice(device_id));
  if (Get().cublas_handle_) {
  	CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
  }
  if (Get().curand_generator_) {
  	CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  }
  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_, generate_seed()));
}

bool Caffe::CheckDevice(const int device_id) {
	bool result = ((cudaSuccess == cudaSetDevice(device_id))
									&& (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  cudaGetLastError();
  return result;
}

int Caffe::FindDevice(const int start_id) {
	int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) {
    	return i;
    }
  }
  return -1;
}

//调用cublas的状态信息
const char* cublasGetErrorString(cublasStatus_t status) {
	switch (status) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
	case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
	case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
	}
	return "Unknown cublas status";
}

//调用curand的状态信息
const char* curandGetErrorString(curandStatus_t status) {
	switch (status) {
	case CURAND_STATUS_SUCCESS:
		return "CURAND_STATUS_SUCCESS";
	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";
	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	}
	return "Unknown curand status";
}

#endif  //!CPU_ONLY
}       //namespace caffe

