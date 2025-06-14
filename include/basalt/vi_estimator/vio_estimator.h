/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <atomic>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/utils/imu_types.h>
#include <basalt/utils/vis_matrices.h>
#include <basalt/linearization/landmark_block.hpp>

namespace basalt {

struct VioVisualizationData {
  using Ptr = std::shared_ptr<VioVisualizationData>;
  using UIMAT = vis::UIMAT;
  using UIJacobians = vis::UIJacobians;
  using UIHessians = vis::UIHessians;
  static constexpr int UIMAT_COUNT_J = vis::UIMAT_COUNT_J;
  static constexpr int UIMAT_COUNT_H = vis::UIMAT_COUNT_H;

  int64_t t_ns;

  Eigen::aligned_map<int64_t, Sophus::SE3d> states;
  Eigen::aligned_map<int64_t, Sophus::SE3d> frames;
  Eigen::aligned_map<int64_t, Sophus::SE3d> ltframes;  // Poses of long-term keyframes
  Eigen::aligned_map<int64_t, size_t> frame_idx{};
  Eigen::aligned_map<int64_t, size_t> keyframed_idx{};
  Eigen::aligned_map<int64_t, size_t> marginalized_idx{};

  Eigen::aligned_vector<Eigen::Vector3d> points;
  std::vector<int> point_ids;

  OpticalFlowResult::Ptr opt_flow_res;

  std::shared_ptr<std::vector<Eigen::aligned_vector<Eigen::Vector4d>>> projections;

  // Indices in Jr and Hb fields
  UIJacobians Jr[UIMAT_COUNT_J];
  UIHessians Hb[UIMAT_COUNT_H];

  UIJacobians& getj(UIMAT u) { return Jr[(int)u]; };
  UIHessians& geth(UIMAT u) { return Hb[(int)u - (int)UIMAT::HB]; };

  void invalidate_mat_imgs() {
    for (UIJacobians& j : Jr) j.img = nullptr;
    for (UIHessians& h : Hb) h.img = nullptr;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class VioEstimatorBase {
 public:
  typedef std::shared_ptr<VioEstimatorBase> Ptr;

  VioEstimatorBase() {
    vision_data_queue.set_capacity(10);
    imu_data_queue.set_capacity(300);
    last_processed_t_ns = 0;
    finished = false;
  }

  std::atomic<int64_t> last_processed_t_ns;
  std::atomic<bool> finished;

  VioVisualizationData::Ptr visual_data;

  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr> vision_data_queue;
  tbb::concurrent_bounded_queue<ImuData<double>::Ptr> imu_data_queue;

  tbb::concurrent_bounded_queue<PoseVelBiasState<double>::Ptr>* out_state_queue = nullptr;
  tbb::concurrent_bounded_queue<MargData::Ptr>* out_marg_queue = nullptr;
  tbb::concurrent_bounded_queue<VioVisualizationData::Ptr>* out_vis_queue = nullptr;

  tbb::concurrent_queue<double>* opt_flow_depth_guess_queue = nullptr;
  tbb::concurrent_queue<PoseVelBiasState<double>::Ptr>* opt_flow_state_queue = nullptr;
  tbb::concurrent_queue<LandmarkBundle::Ptr>* opt_flow_lm_bundle_queue = nullptr;
  tbb::concurrent_queue<Masks>* opt_flow_masks_queue = nullptr;

  virtual void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i, const Eigen::Vector3d& vel_w_i,
                          const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) = 0;

  virtual void initialize(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) = 0;

  virtual void maybe_join() = 0;

  virtual inline void drain_input_queues() {
    // Input threads should abort when vio is finished, but might be stuck in
    // full push to full queue. So this can help to drain queues after joining
    // the processing thread.
    while (!imu_data_queue.empty()) {
      ImuData<double>::Ptr d;
      imu_data_queue.pop(d);
    }
    while (!vision_data_queue.empty()) {
      OpticalFlowResult::Ptr d;
      vision_data_queue.pop(d);
    }
  }

  virtual void scheduleResetState(){};
  virtual void takeLongTermKeyframe(){};

  virtual inline void debug_finalize() {}

  virtual Sophus::SE3d getT_w_i_init() = 0;

  // Legacy functions. Should not be used in the new code.
  virtual void setMaxStates(size_t val) = 0;
  virtual void setMaxKfs(size_t val) = 0;

  virtual void addIMUToQueue(const ImuData<double>::Ptr& data) = 0;
  virtual void addVisionToQueue(const OpticalFlowResult::Ptr& data) = 0;
};

class VioEstimatorFactory {
 public:
  static VioEstimatorBase::Ptr getVioEstimator(const VioConfig& config, const Calibration<double>& cam,
                                               const Eigen::Vector3d& g, bool use_imu, bool use_double);
};

double alignSVD(const std::vector<int64_t>& filter_t_ns, const Eigen::aligned_vector<Eigen::Vector3d>& filter_t_w_i,
                const std::vector<int64_t>& gt_t_ns, Eigen::aligned_vector<Eigen::Vector3d>& gt_t_w_i);

int associate(const std::vector<int64_t>& filter_t_ns,                  //
              const Eigen::aligned_vector<Sophus::SE3d>& filter_T_w_i,  //
              const std::vector<int64_t>& gt_t_ns,                      //
              const Eigen::aligned_vector<Sophus::SE3d>& gt_T_w_i,      //
              Eigen::Matrix<int64_t, Eigen::Dynamic, 1>& out_ts,        //
              Eigen::Matrix<float, 3, Eigen::Dynamic>& out_est_xyz,
              Eigen::Matrix<float, 3, Eigen::Dynamic>& out_ref_xyz,
              Eigen::Matrix<float, 4, Eigen::Dynamic>& out_est_quat,
              Eigen::Matrix<float, 4, Eigen::Dynamic>& out_ref_quat);

Eigen::Matrix4f get_alignment(const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& est_xyz,
                              const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& ref_xyz,  //
                              int i, int j);

float compute_ate(const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& est_xyz,
                  const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& ref_xyz,  //
                  const Eigen::Ref<Eigen::Matrix4f>& T_ref_est_mat,                          //
                  int i, int j);

float compute_rte(const Eigen::Ref<const Eigen::Matrix<int64_t, Eigen::Dynamic, 1>>& est_ts,  //
                  const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& est_xyz,   //
                  const Eigen::Ref<const Eigen::Matrix<float, 4, Eigen::Dynamic>>& est_quat,  //
                  const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& ref_xyz,   //
                  const Eigen::Ref<const Eigen::Matrix<float, 4, Eigen::Dynamic>>& ref_quat,  //
                  Eigen::Matrix<int64_t, Eigen::Dynamic, 1>& out_ts,                          //
                  Eigen::Matrix<float, Eigen::Dynamic, 1>& out_residuals,                     //
                  int i, int j, int delta = 6);

}  // namespace basalt
