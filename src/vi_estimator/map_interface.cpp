// map_interface.cpp
#include <basalt/vi_estimator/map_database.h>
#include <basalt/vi_estimator/map_interface.h>

namespace basalt {

void WriteMapStampMsg::execute(MapDatabase& db) { db.write_map_stamp(map_stamp); }

void WriteMapUpdateMsg::execute(MapDatabase& db) { db.write_map_update(loop_closing_result); }

void WriteMapMargMsg::execute(MapDatabase& db) { db.write_map_marg(keyframes_to_marg); }

void ReadCovisibilityReqMsg::execute(MapDatabase& db) { db.read_covisibility_req(keypoints); }

void Read3dPointsReqMsg::execute(MapDatabase& db) { db.read_3d_points_req(keyframe, neighbors_num); }

}  // namespace basalt
