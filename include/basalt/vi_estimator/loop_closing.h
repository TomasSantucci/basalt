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
#include <basalt/utils/nfr.h>
#include <basalt/utils/sync_utils.h>
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

struct LCTimeStats {
  FrameId timestamp = -1;
  double hash_bow_index_time = 0.0;

  double loop_detection_time = 0.0;

  double hash_bow_search_time = 0.0;
  size_t hash_bow_num_candidates = 0;
  size_t valid_candidate_number = 0;

  double initial_match_time = 0.0;
  double landmarks_request_time = 0.0;
  double island_match_time = 0.0;
  double geometric_verification_time = 0.0;
  double reprojection_time = 0.0;
  double reprojected_geometric_verification_time = 0.0;
  std::vector<double> island_verification_time;

  bool loop_detected = false;

  double loop_closure_time = 0.0;
};

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint3d {
  int id_begin;
  int id_end;

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

using MapOfPoses = std::map<int, Pose3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Pose3d>>>;
using VectorOfConstraints = std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d>>;

struct LoopClosingVisualizationData {
  using Ptr = std::shared_ptr<LoopClosingVisualizationData>;
  using Vec2 = Eigen::Matrix<float, 2, 1>;
  using SE3 = Sophus::SE3<float>;

  int64_t t_ns;

  // a vector of candidate keyframe ids
  std::vector<FrameId> candidate_kfs;

  // a vector of points for each camera
  std::vector<Eigen::aligned_vector<Vec2>> current_keypoints;
  // a map where the key is the tcid of the candidate keyframe and the value is the vector of points
  std::unordered_map<TimeCamId, Eigen::aligned_vector<Vec2>> candidate_keypoints;

  // a map where the key is the tcid of the candidate keyframe and the value is the vector of matches
  std::vector<std::vector<Eigen::aligned_unordered_map<TimeCamId, Eigen::aligned_vector<std::pair<Vec2, Vec2>>>>>
      matches;
  std::vector<std::vector<Eigen::aligned_unordered_map<TimeCamId, Eigen::aligned_vector<std::pair<Vec2, Vec2>>>>>
      inlier_matches;

  // a vector of islands, where each island is a vector of keyframe ids
  std::vector<std::vector<FrameId>> islands;

  std::vector<std::vector<Eigen::aligned_unordered_map<KeypointId, Vec2>>> reprojected_keypoints;
  std::vector<std::vector<std::unordered_map<KeypointId, Eigen::aligned_vector<Vec2>>>> redetected_keypoints;

  std::vector<std::vector<Eigen::aligned_vector<Vec2>>> rematched_keypoints;

  // for each island, the landmarks used for gP3P
  std::vector<Eigen::aligned_vector<Eigen::Vector3d>> landmarks;
  std::vector<std::vector<LandmarkId>> landmark_ids;

  // add the matches that remain after removing the redundant ones for gP3P

  SE3 keyframe_pose;
  std::unordered_map<FrameId, SE3> corrected_pose;  // should change this to be FrameId, for the IMU pose
  std::unordered_map<FrameId, SE3> candidate_pose;  // should change this to be FrameId, for the IMU pose

  std::vector<SE3> corrected_loop_poses;
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
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(t_ab_measured, sqrt_information);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

struct MatchSource {
  FrameId candidate_kf;
  size_t pair_idx;
};

struct KeyframesMatch {
  CamId source_cam;
  CamId target_cam;
  KeypointId source_kpt_id;
  KeypointId target_kpt_id;
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

  tbb::concurrent_bounded_queue<LoopClosingVisualizationData::Ptr>* out_lc_vis_queue = nullptr;
  tbb::concurrent_bounded_queue<std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>>> in_map_res_queue;
  tbb::concurrent_bounded_queue<ReadMessage::Ptr>* out_map_req_queue = nullptr;
  tbb::concurrent_bounded_queue<WriteMessage::Ptr>* out_map_update_queue = nullptr;
  tbb::concurrent_bounded_queue<
      std::shared_ptr<std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Eigen::Matrix<double, 3, 1>>>>>
      in_map_3d_points_queue;

  tbb::concurrent_bounded_queue<LoopClosingInput::Ptr> in_optical_flow_queue;

  SyncState* sync_hashbow_index = nullptr;
  SyncState* sync_vio_finished = nullptr;
  SyncState* sync_lc_finished = nullptr;
  bool deterministic;

 private:
  void updateHashBowDatabase(const LoopClosingInput::Ptr& optical_flow_res);

  bool computeAbsolutePose(const TimeCamId& last_kf_tcid, const std::vector<Landmark<Scalar>>& last_kf_landmarks,
                           const std::vector<Landmark<Scalar>>& candidate_kf_landmarks,
                           const std::vector<std::pair<int, int>>& matches,
                           std::vector<std::pair<int, int>>& inlier_matches, Sophus::SE3d& absolute_pose);

  bool runLoopClosure(const LoopClosingInput::Ptr& optical_flow_res, TimeCamId& best_candidate_tcid,
                      Sophus::SE3f& best_corrected_pose, std::vector<FrameId>& best_island);

  bool closeLoop(const FrameId curr_kf_id, const TimeCamId& best_candidate_tcid,
                 const Sophus::SE3f& best_corrected_pose);

  void query_hashbow_database(FrameId curr_kf_id, HashBowVector& bow_vector, std::vector<FrameId>& results);

  void discard_covisible_keyframes(const FrameId& curr_kf_id, std::vector<FrameId>& candidate_kfs);

  bool redetect_kpts(const Vec2& center_kpt, std::bitset<256>& center_kpt_descriptor,
                     Eigen::aligned_vector<Vec2>& redetected_kpts, const basalt::ManagedImage<uint16_t>& img,
                     Vec2& best_keypoint_pos);

  void reproject_landmarks(const Sophus::SE3d& absolute_pose, FrameId candidate_kf,
                           Eigen::aligned_unordered_map<KeypointId, Vec2>& reprojected_keypoints, size_t cam_id,
                           const Eigen::aligned_map<LandmarkId, Vec3d>& points_3d);

  void filter_matches(std::vector<FrameId>& kfs_island,
                      std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches);

  bool are_covisible(const FrameId& kf1_id, const FrameId& kf2_id);

  void match_keyframe(const FrameId& curr_kf_id,
                      const std::vector<Eigen::aligned_unordered_map<KeypointId, Keypoint>>& curr_kf_kpts,
                      const std::vector<std::vector<KeypointId>>& candidate_kf_kpts, const FrameId& candidate_kf_id,
                      std::vector<KeyframesMatch>& matches);

  void get_neighboring_keyframes(const FrameId& kf_id, int neighbors_num, std::vector<FrameId>& neighboring_kfs);

  void match_candidate_keyframes(const FrameId& curr_kf_id,
                                 const std::vector<Eigen::aligned_unordered_map<KeypointId, Keypoint>>& curr_kf_kpts,
                                 std::vector<FrameId>& candidate_kfs,
                                 std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches);

  size_t computeAbsolutePoseMultiCam(
      const FrameId& last_kf_id,                     // id of the viewpoint we're solving for
      const std::vector<FrameId>& candidate_kf_ids,  // ids of the candidate keyframes
      const std::vector<Eigen::aligned_unordered_map<KeypointId, Keypoint>>&
          last_kf_kpts,  // keypoints of the last kf, per camera
      const std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& points_3d,
      const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches,
      std::unordered_map<FrameId, std::vector<KeyframesMatch>>& inlier_matches, Sophus::SE3d& absolute_pose);

  void validateLoopCandidates(
      const TimeCamId& curr_kf_tcid, const std::vector<Landmark<Scalar>>& curr_kf_landmarks,
      const std::unordered_map<TimeCamId, std::vector<Landmark<Scalar>>>& candidate_kf_landmarks_map,
      const std::vector<TimeCamId>& loop_candidates,
      const std::unordered_map<TimeCamId, std::vector<std::pair<int, int>>>& matches_map,
      std::vector<TimeCamId>& validated_candidates, std::vector<Sophus::SE3f>& corrected_poses);

  void buildPoseGraph(const TimeCamId& curr_kf_tcid, const TimeCamId& best_candidate_tcid,
                      const Sophus::SE3f& best_corrected_pose, std::vector<Sophus::SE3f>& poses,
                      std::vector<Sophus::SE3f>& relative_poses);

  void buildCeresParams(const std::vector<Sophus::SE3f>& poses, const std::vector<Sophus::SE3f>& relative_poses,
                        MapOfPoses& map_of_poses, VectorOfConstraints& constraints);

  void restorePosesFromCeres(const MapOfPoses& map_of_poses, std::vector<Sophus::SE3f>& poses);

  void updateMap(const std::vector<Sophus::SE3f>& poses, const TimeCamId& best_candidate_tcid);

  void buildOptimizationProblem(const VectorOfConstraints& constraints, MapOfPoses* map_of_poses,
                                ceres::Problem* problem);

  bool solveOptimizationProblem(ceres::Problem* problem);

  VioConfig config;
  basalt::Calibration<double> calib;

  std::map<TimeCamId, std::unordered_map<KeypointId, std::bitset<256>>> kpt_descriptors;
  std::unordered_map<TimeCamId, std::vector<KeypointId>> kf_landmarks;
  std::unordered_map<TimeCamId, std::unordered_map<KeypointId, Vec2>> kpts_positions;
  std::shared_ptr<HashBow<256>> hash_bow_database;

  LandmarkDatabase<Scalar>::Ptr map;
  std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>> loop_kfs_poses;
  LCTimeStats lc_time_stats;

  bool close_loop = false;

  std::shared_ptr<std::thread> processing_thread;
  LoopClosingVisualizationData::Ptr loop_closing_visualization_data;
};
}  // namespace basalt
