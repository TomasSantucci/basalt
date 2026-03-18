#pragma once

#include <basalt/vi_estimator/covisibility_graph.h>
#include <basalt/vi_estimator/landmark_database.h>
#include "basalt/utils/common_types.h"

namespace basalt {

struct LoopClosingResult {
  using Vec2 = Eigen::Matrix<float, 2, 1>;
  using Ptr = std::shared_ptr<LoopClosingResult>;

  std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>> keyframe_poses;
  Sophus::SE3f current_kf_corrected_pose;

  FrameId candidate_kf_id, current_kf_id;

  std::unordered_map<LandmarkId, LandmarkId> lm_fusions;
  std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Vec2>> curr_kf_obs;
};

struct MapStamp {
  typedef std::shared_ptr<MapStamp> Ptr;

  int64_t t_ns;
  typename LandmarkDatabase<float>::Ptr lmdb;
};

struct MarginalizationStamp {
  using Ptr = std::shared_ptr<MarginalizationStamp>;

  std::set<FrameId> keyframe_ids;
};

struct CovisibilityRequest {
  using Ptr = std::shared_ptr<CovisibilityRequest>;

  std::vector<KeypointId> keypoints;
};

struct IslandRequest {
  using Ptr = std::shared_ptr<IslandRequest>;

  FrameId keyframe;
  size_t neighbors_num;
};

struct LoopClosureDecision {
  using Ptr = std::shared_ptr<LoopClosureDecision>;

  std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>> keyframe_poses;
  CovisibilityGraph covisibility_graph;
  std::set<FrameId> active_keyframes;

  bool close_loop;
};

struct IslandResponse {
  using Ptr = std::shared_ptr<IslandResponse>;

  std::vector<FrameId> keyframes;
  std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Eigen::Matrix<double, 3, 1>>> landmarks_3d_map;
};

struct OptimizedPosesUpdate {
  using Ptr = std::shared_ptr<OptimizedPosesUpdate>;

  std::shared_ptr<Eigen::aligned_map<int64_t, Sophus::SE3f>> keyframe_poses;
};

struct StopMsg {};

using MapWriteMessage = std::variant<MapStamp::Ptr, MarginalizationStamp::Ptr, LoopClosingResult::Ptr, StopMsg>;
using MapReadMessage = std::variant<CovisibilityRequest::Ptr, IslandRequest::Ptr, StopMsg>;

}  // namespace basalt
