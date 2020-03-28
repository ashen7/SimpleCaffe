//
// Created by yipeng on 2020/3/27.
//
#ifndef SIMPLE_CAFFE_LAYER_FACTORY_HPP_
#define SIMPLE_CAFFE_LAYER_FACTORY_HPP_

#include <vector>
#include <map>
#include <string>
#include <functional>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//类前置声明
template <typename Dtype>
class LayerInterface;

//添加层和初始化层
template <typename Dtype>
class LayerRegistry {
 public:
	//定义一个函数指针类型 Creator
	//typedef shared_ptr<LayerInterface<Dtype>>	(*Creator)(const LayerParameter&);
	typedef std::function<shared_ptr<LayerInterface<Dtype>>(const LayerParameter&)> Creator;
	typedef std::map<string, Creator> CreatorRegistry;

	//得到map对象 全局变量 前面加g
	static CreatorRegistry& Registry() {
		static CreatorRegistry g_registry_;
		return g_registry_;
	}

	//注册一个层 添加一个Creator到map里面
	static void AddCreator(const string& type, Creator creator) {
		CreatorRegistry& registry = Registry();
		CHECK_EQ(registry.count(type), 0) << "Layer type "
				<< type << " already registered";
		registry[type] = creator;
	}

	//传入层参数 调用已经注册的指定层的函数 然后返回层接口的对象
	static shared_ptr<LayerInterface<Dtype>> CreateLayer(const LayerParameter& param) {
		if (Caffe::root_solver()) {
			LOG(INFO) << "Creating layer " << param.name();
		}
		const string& type = param.type();
		CreatorRegistry& registry = Registry();
		//如果没有注册过这个层 就退出
		CHECK_EQ(registry.count(type), 1) << "Unknown layer type: "
				<< type << " (known types: " << LayerTypeListString() << ")";

		return registry[type](param);
	}

	//得到所有注册了的层的类型
	static vector<string> LayerTypeList() {
		CreatorRegistry& registry = Registry();
		vector<string> layer_types;
		for (const auto& creator_registry : registry) {
			layer_types.push_back(creator_registry.first);
		}

		return layer_types;
	}

 private:
	LayerRegistry() {}

	//得到注册了的层类型字符串
	static string LayerTypeListString() {
		vector<string> layer_types = LayerTypeList();
		string layer_types_str;
		int n = 0;
		for (const auto& layer_type : layer_types) {
			if (n != 0) {
				layer_types_str += ", ";
			}
			layer_types_str += layer_type;
			n++;
		}

		return layer_types_str;
	}
};     //class LayerRegistry

//添加一个层
template <typename Dtype>
class LayerRegisterer {
 public:
	LayerRegisterer(const string& type,
	                std::function<shared_ptr<LayerInterface<Dtype>>(const LayerParameter&)> creator) {
		LayerRegistry<Dtype>::AddCreator(type, creator);
	}
};     //class LayerRegisterer

//#表示 将宏参数字符串化
#define REGISTER_LAYER_CREATOR(type, creator) \
	static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>); \
	static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)

#define REGISTER_LAYER_CLASS(type) \
	template <typename Dtype> \
	shared_ptr<LayerInterface<Dtype>> Creator_##type##Layer(const LayerParameter& param) \
	{ \
		return shared_ptr <LayerInterface<Dtype>>(new type##Layer<Dtype>(param)); \
	} \
	REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}      //namespace caffe

#endif //SIMPLE_CAFFE_LAYER_FACTORY_HPP_
