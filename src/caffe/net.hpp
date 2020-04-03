//
// Created by yipeng on 2020/4/1.
//
#ifndef SIMPLE_CAFFE_NET_HPP_
#define SIMPLE_CAFFE_NET_HPP_

#include <vector>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "caffe/tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//网络(net): 由多个层组成的DAG有向无环图 用Proto NetParameter来构造
template <typename Dtype>
class Net {
 public:
	explicit Net(const NetParameter& param);
	explicit Net(const string& param_file, Phase phase,
		           const int level = 0, const vector<string>* stages = nullptr);
	virtual ~Net() {}

	//用proto网络参数来初始化net
	void Init(const NetParameter& param);

	//前向计算
	const vector<Tensor<Dtype>*>& Forward(Dtype* loss = nullptr);

	Dtype ForwardFromTo(int start, int end);
	Dtype ForwardFrom(int start);
	Dtype ForwardTo(int end);

	//清空梯度
	void ClearParamDiffs();

	//反向计算
	void Backward();
	void BackwardFromTo(int start, int end);
	void BackwardFrom(int start);
	void BackwardTo(int end);

	//reshape 所有层用输入size得到输出size
	void Reshape();

	//前向计算 + 反向计算
	Dtype ForwardBackward() {
		Dtype loss;
		Forward(&loss);
		Backward();
		return loss;
	}

	//用梯度值来更新网络每层权重
	void Update();

	void ShareWeights();

	void ShareTrainedLayersWith(const Net* other);

	void CopyTrainedLayersFrom(const NetParameter& param);
	void CopyTrainedLayersFrom(const string& trained_filename);
	void CopyTrainedLayersFromBinaryProto(const string& trained_filename);
	void CopyTrainedLayersFromHDF5(const string& trained_filename);

	void ToProto(NetParameter* param, bool write_diff = false) const;
	void ToHDF5(const string& filename, bool write_diff = false) const;

	inline const string& name() const { return name_; }  //网络的名字
	inline const vector<string>& layer_names() const { return layer_names_; } //层的名字
	inline const vector<string>& tensor_names() const { return tensor_names_; } //tensor的名字
	inline const vector<shared_ptr<Tensor<Dtype>>>& tensors() const { return tensors_; }
	inline const vector<shared_ptr<LayerInterface<Dtype>>>& layers() const { return layers_; }
	inline Phase phase() const { return phase_; }
	inline const vector<vector<Tensor<Dtype>*>>& bottom_list() const { return bottom_list_; }




 protected:

	//层参数
	string name_;   //网络的名字
	Phase phase_;   //阶段 Train/Test
	vector<shared_ptr<LayerInterface<Dtype>>> layers_;  //层
	vector<string> layer_names_;
	map<string, int> layer_names_index_;
	vector<bool> layer_need_backward_;

	//存储层之间的中间结果
	vector<shared_ptr<Tensor<Dtype>>> tensors_;
	vector<string> tensor_names_;
	map<string, int> tensor_names_index_;
	vector<bool> tensor_need_backward_;

	//存放每个层的输入tensor
	vector<vector<Tensor<Dtype>*>> bottom_list_;
	vector<vector<int>> bottom_index_list_;
	vector<vector<bool>> bottom_need_backward_;

	//存放每个层的输出tensor
	vector<vector<Tensor<Dtype>*>> top_list_;
	vector<vector<int>> top_index_list_;

	vector<Dtype> tensor_loss_weights_;
	vector<vector<int>> weights_index_list_;

	DISABLE_COPY_AND_ASSIGN(Net);
};     //class Net

}      //namespace caffe

#endif //SIMPLE_CAFFE_NET_HPP_
