//
// Created by yipeng on 2020/4/2.
//
#ifndef SIMPLE_CAFFE_SOLVER_HPP_
#define SIMPLE_CAFFE_SOLVER_HPP_

#include <string>
#include <vector>
#include <functional>

#include "caffe/net.hpp"


namespace caffe {

	//ctrl-c退出程序时 的动作
	namespace SolverAction {
		enum ExitAction {
			NONE = 0,      //直接退出
			STOP = 1,      //停止训练
			SNAPSHOT = 2,  //保存模型 继续训练
		};
	}

typedef std::function<SolverAction::ExitAction()>	ActionCallback;

//执行训练/测试的接口 操作Net
template <typename Dtype>
class Solver {
 public:
	explicit Solver(const SolverParameter& param);
	explicit Solver(const string& param_file);
	void Init(const SolverParameter& param);
	void InitTrainNet();
	void InitTestNets();

	void SetActionFunction(ActionCallback func);
	SolverAction::ExitAction GetRequestedAction();

	virtual void Solve(const char* resume_file = nullptr);
	inline void Solve(const string& resume_file) { Solve(resume_file.c_str()); }
	void Step(int iters);


 protected:
	string SnapshotFilename(const string& extension);
	string SnapshotToBinaryProto();
	string SnapshotToHDF5();
	//测试
	void TestAll();
	void Test(const int test_net_index = 0);
	virtual void SnapshotSolverState(const string& model_filename) = 0;
	virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
	virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
	void DisplayOutputTensors(const int net_index);
	void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

	SolverParameter param_;
	int iter_;
	int current_step_;
	shared_ptr<Net<Dtype>> net_;
	vector<shared_ptr<Net<Dtype>>> test_nets_;
	vector<Dtype> losses_;
	Dtype smoothed_loss_;

	ActionCallback action_request_function_;

	bool requested_early_exit_;

	DISABLE_COPY_AND_ASSIGN(Solver);
};     //class Solver

}      //namespace caffe

#endif //SIMPLE_CAFFE_SOLVER_HPP_
