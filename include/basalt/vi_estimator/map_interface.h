#pragma once

#include <basalt/vi_estimator/landmark_database.h>
#include "basalt/utils/common_types.h"

namespace basalt {

class MapDatabase;

struct MapStamp {
  typedef std::shared_ptr<MapStamp> Ptr;

  int64_t t_ns;
  typename LandmarkDatabase<float>::Ptr lmdb;
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
  std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>> keyframe_poses;
  FrameId candidate_kf_id;
  FrameId curr_kf_id;
  std::unordered_map<LandmarkId, LandmarkId> lm_fusions;
  std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Eigen::Matrix<float, 2, 1>>> curr_lc_obs;
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

struct ReadMapReqMsg : ReadMessage {
  FrameId frame_id;
  void execute(MapDatabase& db) override;
};

}  // namespace basalt
