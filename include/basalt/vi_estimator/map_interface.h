#pragma once

#include <basalt/vi_estimator/landmark_database.h>
#include "basalt/utils/common_types.h"

namespace basalt {

class MapDatabase;

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
  std::set<FrameId> keyframes_to_marg;
};

struct WriteMessage {
  typedef std::shared_ptr<WriteMessage> Ptr;

  virtual ~WriteMessage() = default;
  virtual void execute(MapDatabase& db) = 0;
};

struct WriteMapStampMsg : WriteMessage {
  MapStamp::Ptr map_stamp;
  void execute(MapDatabase& db) override;
};

struct WriteMapUpdateMsg : WriteMessage {
  LoopClosingResult::Ptr loop_closing_result;
  void execute(MapDatabase& db) override;
};

struct WriteMapMargMsg : WriteMessage {
  std::set<FrameId> keyframes_to_marg;
  void execute(MapDatabase& db) override;
};

struct ReadMessage {
  typedef std::shared_ptr<ReadMessage> Ptr;

  virtual ~ReadMessage() = default;
  virtual void execute(MapDatabase& db) = 0;
};

struct ReadCovisibilityReqMsg : ReadMessage {
  std::shared_ptr<std::vector<KeypointId>> keypoints;
  void execute(MapDatabase& db) override;
};

struct Read3dPointsReqMsg : ReadMessage {
  FrameId keyframe;
  size_t neighbors_num;
  void execute(MapDatabase& db) override;
};

}  // namespace basalt
