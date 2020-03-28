//
// Created by yipeng on 2020/3/27.
//
#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#endif

namespace caffe {

//relu层的注册函数 通过engine来判断用那个接口
template <typename Dtype>
shared_ptr<LayerInterface<Dtype>> GetReLULayer(const LayerParameter& param) {
	ReLUParameter_Engine engine = param.relu_param().engine();
	if (engine == ReLUParameter_Engine_DEFAULT) {
		engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
		engine = ReLUParameter_Engine_CUDNN;
#endif
	}

	if (engine == ReLUParameter_Engine_CAFFE) {
		return shared_ptr<LayerInterface<Dtype>>(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
	} else if (engine == ReLUParameter_Engine_CUDNN) {
		return shared_ptr<LayerInterface<Dtype>>(new CuDNNReLULayer<Dtype>(param));
#endif
	}	else {
		LOG(FATAL) << "Layer " << param.name() << " has unknown engine";
		throw;  //避免丢失返回警告
	}
}

//注册relu层
REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

//sigmoid层的注册函数 通过engine来判断用那个接口
template <typename Dtype>
shared_ptr<LayerInterface<Dtype>> GetSigmoidLayer(const LayerParameter& param) {
	SigmoidParameter_Engine engine = param.sigmoid_param().engine();
	if (engine == SigmoidParameter_Engine_DEFAULT) {
		engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
		engine = SigmoidParameter_Engine_CUDNN;
#endif
	}

	if (engine == SigmoidParameter_Engine_CAFFE) {
		return shared_ptr<LayerInterface<Dtype>>(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
	} else if (engine == SigmoidParameter_Engine_CUDNN) {
		return shared_ptr<LayerInterface<Dtype>>(new CuDNNSigmoidLayer<Dtype>(param));
#endif
	} else {
		LOG(FATAL) << "Layer " << param.name() << " has unknown engine";
		throw;  //避免丢失返回警告
	}
}

//注册sigmoid层
REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

}      //namespace caffe