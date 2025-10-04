#pragma once

#include <basalt/vi_estimator/landmark_database.h>

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
  LandmarkDatabase<float>::Ptr map_update;
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

struct ReadMapReqMsg : ReadMessage {
  bool req;
  void execute(MapDatabase& db) override;
};

}  // namespace basalt
