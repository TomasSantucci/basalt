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

#include <memory>
#include <thread>

#include <basalt/optical_flow/optical_flow.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <unordered_map>
#include "basalt/vi_estimator/map_interface.h"

#include <basalt/hash_bow/hash_bow.h>
#include <basalt/utils/common_types.h>
#include <basalt/utils/lc_time_utils.h>
#include <basalt/utils/nfr.h>
#include <basalt/utils/sync_utils.h>
#include <basalt/vi_estimator/map_database.h>
#include <basalt/vi_estimator/sc_ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace basalt {

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  // The name of the data type in the g2o file format.
  static std::string name() { return "VERTEX_SE3:QUAT"; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint3d {
  FrameId id_begin;
  FrameId id_end;

  // The transformation that represents the pose of the end frame E w.r.t. the
  // begin frame B. In other words, it transforms a vector in the E frame to
  // the B frame.
  Pose3d t_be;

  // The inverse of the covariance matrix for the measurement. The order of the
  // entries are x, y, z, delta orientation.
  Eigen::Matrix<double, 6, 6> information;

  // The name of the data type in the g2o file format.
  static std::string name() { return "EDGE_SE3:QUAT"; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using MapOfPoses =
    std::map<FrameId, Pose3d, std::less<FrameId>, Eigen::aligned_allocator<std::pair<const FrameId, Pose3d>>>;
using VectorOfConstraints = std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d>>;

struct MatchSource {
  FrameId candidate_kf;
  size_t pair_idx;
};

struct KeyframesMatch {
  CamId current_kf_cam;
  CamId candidate_kf_cam;
  KeypointId current_kf_kpt_id;
  KeypointId candidate_kf_kpt_id;
};

struct LoopClosingVisualizationData {
  using Ptr = std::shared_ptr<LoopClosingVisualizationData>;
  using Vec2 = Eigen::Matrix<float, 2, 1>;
  using SE3 = Sophus::SE3<float>;

  int64_t t_ns;

  // Keypoints
  std::vector<Keypoints> current_keypoints;
  std::unordered_map<TimeCamId, Keypoints> candidate_keypoints;

  // Matches
  std::unordered_map<FrameId, std::vector<KeyframesMatch>> matches;
  std::unordered_map<FrameId, std::vector<KeyframesMatch>> inlier_matches;

  // The island
  std::vector<FrameId> island;

  // Reprojection data
  std::vector<Keypoints> reprojected_keypoints;
  std::vector<std::unordered_map<KeypointId, Eigen::aligned_vector<Vec2>>> redetected_keypoints;
  std::vector<Eigen::aligned_vector<Vec2>> rematched_keypoints;

  // Poses
  SE3 candidate_corrected_pose;

  // Whether the loop was closed
  bool loop_closed = false;
};

class PoseGraph3dErrorTerm {
 public:
  PoseGraph3dErrorTerm(Pose3d t_ab_measured, Eigen::Matrix<double, 6, 6> sqrt_information)
      : t_ab_measured_(std::move(t_ab_measured)), sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T* const p_a_ptr, const T* const q_a_ptr, const T* const p_b_ptr, const T* const q_b_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q = t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Pose3d& t_ab_measured, const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
        new PoseGraph3dErrorTerm(t_ab_measured, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

class LoopClosing {
 public:
  using Scalar = float;

  using Ptr = std::shared_ptr<LoopClosing>;
  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Vec4d = Eigen::Matrix<double, 4, 1>;
  using Mat4d = Eigen::Matrix<double, 4, 4>;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4f = Eigen::Matrix<float, 4, 1>;

  LoopClosing(const VioConfig& config, const basalt::Calibration<double>& calib);

  ~LoopClosing() { maybe_join(); }

  void initialize();

  void triggerLoopClosure();

  inline void maybe_join() {
    if (processing_thread) {
      processing_thread->join();
      processing_thread.reset();
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  tbb::concurrent_bounded_queue<MapReadMessage>* out_map_read_queue = nullptr;
  tbb::concurrent_bounded_queue<MapWriteMessage>* out_map_write_queue = nullptr;

  tbb::concurrent_bounded_queue<LoopClosingVisualizationData::Ptr>* out_lc_vis_queue = nullptr;
  tbb::concurrent_bounded_queue<LoopClosureDecision::Ptr> in_lc_dec_res_queue;
  tbb::concurrent_bounded_queue<IslandResponse::Ptr> in_island_res_queue;
  tbb::concurrent_bounded_queue<LoopClosingInput::Ptr> in_optical_flow_queue;

  SyncState* sync_lc_finished = nullptr;
  bool deterministic;

 private:
  void updateHashBowDatabase(const LoopClosingInput::Ptr& optical_flow_res, HashBowVector& bow_vector);

  bool runLoopClosure(const LoopClosingInput::Ptr& optical_flow_res, const HashBowVector& bow_vector,
                      TimeCamId& best_candidate_tcid, Sophus::SE3f& best_corrected_pose,
                      std::vector<FrameId>& best_island);

  bool closeLoop(const FrameId curr_kf_id, const std::vector<FrameId>& best_island,
                 const Sophus::SE3f& best_corrected_pose);

  void query_hashbow_database(FrameId curr_kf_id, const HashBowVector& bow_vector, std::vector<FrameId>& results,
                              std::vector<double>& scores);

  bool redetect_kpts(const Keypoint& center_kpt, std::bitset<256>& center_kpt_descriptor,
                     Eigen::aligned_vector<Vec2>& redetected_kpts, const basalt::ManagedImage<uint16_t>& img,
                     Vec2& best_keypoint_pos);

  void reproject_landmarks(const Sophus::SE3d& absolute_pose, Keypoints& reprojected_keypoints, size_t cam_id,
                           const Eigen::aligned_map<LandmarkId, Vec3d>& points_3d,
                           const std::unordered_set<LandmarkId>& landmarks_to_skip);

  void populateVisualizationData(
      FrameId curr_kf_id, const std::vector<FrameId>& kfs_island,
      const std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& points_3d,
      const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches,
      const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& inlier_matches,
      const std::vector<Keypoints>& curr_kf_kpts, const std::vector<Keypoints>& reprojected_keypoints,
      const std::vector<std::unordered_map<KeypointId, Eigen::aligned_vector<Vec2>>>& redetected_keypoints_map,
      const std::vector<Eigen::aligned_vector<Vec2>>& rematched_keypoints, const Sophus::SE3f& best_corrected_pose,
      const TimeCamId& best_candidate_tcid);

  bool are_covisible(const FrameId& kf1_id, const FrameId& kf2_id);

  void filter_redundant_matches(std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches);

  void match_keyframe(const FrameId& curr_kf_id, const std::vector<Keypoints>& curr_kf_kpts,
                      const std::vector<std::vector<KeypointId>>& candidate_kf_kpts, const FrameId& candidate_kf_id,
                      std::vector<KeyframesMatch>& matches);

  size_t computeAbsolutePoseMultiCam(
      const FrameId& last_kf_id,                     // id of the viewpoint we're solving for
      const std::vector<FrameId>& candidate_kf_ids,  // ids of the candidate keyframes
      const std::vector<Keypoints>& last_kf_kpts,    // keypoints of the last kf, per camera
      const std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& points_3d,
      const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches,
      std::unordered_map<FrameId, std::vector<KeyframesMatch>>& inlier_matches, Sophus::SE3d& absolute_pose);

  void buildPoseGraph(const TimeCamId& curr_kf_tcid, const std::vector<FrameId>& best_island,
                      const Sophus::SE3f& best_corrected_pose, MapOfPoses& map_of_poses,
                      VectorOfConstraints& constraints);

  void restorePosesFromCeres(const MapOfPoses& map_of_poses);

  void buildOptimizationProblem(const VectorOfConstraints& constraints, MapOfPoses* map_of_poses,
                                ceres::Problem* problem);

  bool solveOptimizationProblem(ceres::Problem* problem);

  VioConfig config;
  basalt::Calibration<double> calib;

  std::map<TimeCamId, std::unordered_map<KeypointId, std::bitset<256>>> kpt_descriptors;
  std::unordered_map<TimeCamId, std::vector<KeypointId>> kf_landmarks;
  std::unordered_map<TimeCamId, std::unordered_map<KeypointId, Vec2>> kpts_positions;
  std::shared_ptr<HashBow<256>> hash_bow_database;

  LoopClosureDecision::Ptr loop_closure_decision;
  LCTimeStats lc_time_stats;

  bool close_loop = false;

  std::shared_ptr<std::thread> processing_thread;
  LoopClosingVisualizationData::Ptr loop_closing_visualization_data;
  LoopDetectionResult::Ptr loop_detection_result;
};
}  // namespace basalt
