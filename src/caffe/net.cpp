//
// Created by yipeng on 2020/4/1.
//
#include <vector>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <algorithm>

#ifdef USE_HDF5
#include "hdf5.h"
#endif //USE_HDF5

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {
	Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
                const int level, const vector<string>* stages) {
	NetParameter param;
//	ReadNet
	param.mutable_state()->set_phase(phase);
	if (stages != nullptr) {
		for (int i = 0; i < stages->size(); ++i) {
			param.mutable_state()->add_stage((*stages)[i]);
		}
	}
	param.mutable_state()->set_level(level);
	Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& param) {

}

}     //namespace caffe