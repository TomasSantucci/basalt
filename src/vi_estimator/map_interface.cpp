// map_interface.cpp
#include <basalt/vi_estimator/map_database.h>
#include <basalt/vi_estimator/map_interface.h>

namespace basalt {

void WriteMapStampMsg::execute(MapDatabase& db) { db.write_map_stamp(map_stamp); }

void WriteMapUpdateMsg::execute(MapDatabase& db) { db.write_map_update(map_update); }

void ReadCovisibilityReqMsg::execute(MapDatabase& db) { db.read_covisibility_req(keypoints); }

void Read3dPointsReqMsg::execute(MapDatabase& db) { db.read_3d_points_req(keyframes); }

void ReadMapReqMsg::execute(MapDatabase& db) { db.read_map_req(frame_id); }

}  // namespace basalt
