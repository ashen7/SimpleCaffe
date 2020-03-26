//
// Created by yipeng on 2020/3/26.
//
#ifdef USE_CUDNN

#include "caffe/util/cudnn.hpp"

namespace caffe {
namespace cudnn {

//定义类静态成员
float dataType<float>::zeroval = 0.0;
float dataType<float>::oneval = 1.0;
const void* dataType<float>::zero =
		static_cast<void*>(&dataType<float>::zeroval);
const void* dataType<float>::one =
		static_cast<void*>(&dataType<float>::oneval);

double dataType<double>::zeroval = 0.0;
double dataType<double>::oneval = 1.0;
const void* dataType<double>::zero =
		static_cast<void*>(&dataType<double>::zeroval);
const void* dataType<double>::one =
		static_cast<void*>(&dataType<double>::oneval);

}      //namespace cudnn
}      //namespace caffe
#endif //USE_CUDNN

