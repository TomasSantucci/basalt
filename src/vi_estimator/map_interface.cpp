// map_interface.cpp
#include <basalt/vi_estimator/map_database.h>
#include <basalt/vi_estimator/map_interface.h>

namespace basalt {

void WriteMapStampMsg::execute(MapDatabase& db) { db.write_map_stamp(map_stamp); }

void WriteMapUpdateMsg::execute(MapDatabase& db) {
  db.write_map_update(keyframe_poses, candidate_kf_id, curr_kf_id, lm_fusions, curr_lc_obs);
}

void ReadCovisibilityReqMsg::execute(MapDatabase& db) { db.read_covisibility_req(keypoints); }

void Read3dPointsReqMsg::execute(MapDatabase& db) { db.read_3d_points_req(keyframe, neighbors_num); }

void ReadMapReqMsg::execute(MapDatabase& db) { db.read_map_req(frame_id); }

}  // namespace basalt
