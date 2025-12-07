#include <basalt/vi_estimator/map_database.h>
#include "basalt/utils/common_types.h"
#include "basalt/utils/eigen_utils.hpp"
#include "basalt/utils/keypoints.h"
#include "basalt/vi_estimator/landmark_database.h"
#include "basalt/vi_estimator/map_interface.h"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

namespace basalt {

MapDatabase::MapDatabase(const VioConfig& config, const Calibration<double>& calib) {
  this->config = config;
  this->calib = calib;
  this->map = std::make_shared<LandmarkDatabase<float>>("Persistent Map");
}

void MapDatabase::write_map_stamp(basalt::MapStamp::Ptr& map_stamp) {
  if (map_stamp == nullptr) {
    map->print();
    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_map_update_queue) out_map_update_queue->push(nullptr);
    return;
  }

  std::unique_lock<std::mutex> lock(mutex);
  map->mergeLMDB(map_stamp->lmdb, true);

  if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_STS) {
    std::set<TimeCamId> kfs_to_compute;
    for (auto const& [kf_id, _] : map_stamp->lmdb->getKeyframeObs()) kfs_to_compute.emplace(kf_id);
    computeSpatialDistributions(kfs_to_compute);
  }

  if (out_vis_queue) {
    map_visual_data = std::make_shared<MapDatabaseVisualizationData>();
    map_visual_data->t_ns = map_stamp->t_ns;
    computeMapVisualData();
    out_vis_queue->push(map_visual_data);
  }

  if (sync_map_stamp != nullptr) {
    std::lock_guard<std::mutex> lk(sync_map_stamp->m);
    sync_map_stamp->ready = true;
    sync_map_stamp->cvar.notify_one();
  }

  if (requested_frame_id != -1 && map->keyframeExists(requested_frame_id)) {
    requested_frame_id = -1;

    auto map_msg = std::make_shared<ReadMapReqMsg>();
    map_msg->frame_id = requested_frame_id;
    std::cout << "Fulfilling pending map request for frame id " << requested_frame_id << std::endl;
    read_queue.push(map_msg);
  } else if (requested_frame_id != -1) {
    std::cout << "Still waiting for frame id " << requested_frame_id << " to be added to the map." << std::endl;
  }
}

void MapDatabase::write_map_update(std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>>& map_update) {
  if (map_update == nullptr) return;

  std::unique_lock<std::mutex> lock(mutex);

  map->mergeKeyframesPoses(map_update);

  if (out_map_update_queue) {
    // possibly send the whole map visual data as well
    MapUpdate::Ptr map_update_msg = std::make_shared<basalt::MapUpdate>();
    map_update_msg->keyframe_poses = map->getKeyframes();
    out_map_update_queue->push(map_update_msg);
  }

  if (sync_lc_finished != nullptr) {
    std::lock_guard<std::mutex> lk(sync_lc_finished->m);
    sync_lc_finished->ready = true;
    sync_lc_finished->cvar.notify_one();
  }
}

void MapDatabase::read_covisibility_req(std::shared_ptr<std::vector<KeypointId>>& keypoints_ptr) {
  if (keypoints_ptr == nullptr) {
    return;
  }
  std::vector<KeypointId>& keypoints = *keypoints_ptr;

  std::unique_lock<std::mutex> lock(mutex);
  handleCovisibilityReq(keypoints);
}

void MapDatabase::read_3d_points_req(std::shared_ptr<std::vector<FrameId>>& keyframes) {
  if (keyframes == nullptr) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex);

  auto landmarks_3d_map = std::make_shared<std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>>();

  for (const auto& kf_id : *keyframes) {
    for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
      if (!map->keyframeExists(kf_id)) {
        std::cout << "Keyframe " << kf_id << " does not exist in the map database." << std::endl;
        continue;
      }

      TimeCamId kf_tcid{kf_id, cam_id};
      auto it = map->getKeyframeObs().find(kf_tcid);
      if (it == map->getKeyframeObs().end()) {
        std::cout << "No observations for keyframe " << kf_id << " and cam " << cam_id << std::endl;
        continue;
      }
      std::set<LandmarkId> lm_ids = it->second;

      // TODO: this should be performed by LC
      for (const auto& lm_id : lm_ids) {
        auto lm_pos = map->getLandmark(lm_id);
        int64_t frame_id = lm_pos.host_kf_id.frame_id;
        Sophus::SE3d T_w_i = map->getKeyframePose(frame_id).cast<double>();

        const Sophus::SE3d& T_i_c = calib.T_i_c[lm_pos.host_kf_id.cam_id];
        Mat4d T_w_c = (T_w_i * T_i_c).matrix();

        Vec4d pt_cam = StereographicParam<double>::unproject(lm_pos.direction.cast<double>());
        pt_cam[3] = lm_pos.inv_dist;

        Vec4d pt_w = T_w_c * pt_cam;

        (*landmarks_3d_map)[kf_tcid][lm_id] = (pt_w.template head<3>() / pt_w[3]).template cast<double>();
      }
    }
  }

  if (out_3d_points_res_queue) {
    out_3d_points_res_queue->push(landmarks_3d_map);
  }
}

void MapDatabase::read_map_req(FrameId frame_id) {
  std::unique_lock<std::mutex> lock(mutex);

  if (!map->keyframeExists(frame_id)) {
    std::cout << "Keyframe " << frame_id << " does not exist in the map yet. Deferring map request." << std::endl;
    requested_frame_id = frame_id;
    return;
  }

  auto keyframes_poses_copy = std::make_shared<Eigen::aligned_map<FrameId, Sophus::SE3f>>(map->getKeyframes());
  if (out_map_res_queue) {
    out_map_res_queue->push(keyframes_poses_copy);
  }
}

void MapDatabase::initialize() {
  auto read_func = [&]() {
    std::shared_ptr<ReadMessage> msg;
    while (true) {
      read_queue.pop(msg);
      if (msg == nullptr) break;
      msg->execute(*this);
    }
  };

  auto write_func = [&]() {
    std::shared_ptr<WriteMessage> msg;
    while (true) {
      write_queue.pop(msg);
      if (msg == nullptr) {
        map->print();
        if (out_vis_queue) out_vis_queue->push(nullptr);
        break;
      }
      msg->execute(*this);
    }
  };

  reading_thread.reset(new std::thread(read_func));
  writing_thread.reset(new std::thread(write_func));
}

void MapDatabase::get_map_points(Eigen::aligned_vector<Vec3d>& points, std::vector<int>& ids) {
  points.clear();
  ids.clear();

  for (const auto& tcid_host : map->getHostKfs()) {
    int64_t id = tcid_host.frame_id;
    Sophus::SE3d T_w_i = map->getKeyframePose(id).cast<double>();

    const Sophus::SE3d& T_i_c = calib.T_i_c[tcid_host.cam_id];
    Mat4d T_w_c = (T_w_i * T_i_c).matrix();

    for (const auto& [lm_id, lm_pos] : map->getLandmarksForHostWithIds(tcid_host)) {
      Vec4d pt_cam = StereographicParam<double>::unproject(lm_pos->direction.cast<double>());
      pt_cam[3] = lm_pos->inv_dist;

      Vec4d pt_w = T_w_c * pt_cam;

      points.emplace_back((pt_w.template head<3>() / pt_w[3]).template cast<double>());
      ids.emplace_back(lm_id);
    }
  }
}

Eigen::aligned_map<LandmarkId, Eigen::Matrix<double, 3, 1>> MapDatabase::get_landmarks_3d_pos(
    std::set<LandmarkId> landmarks) {
  Eigen::aligned_map<LandmarkId, Vec3d> landmarks_3d{};

  for (const auto lm_id : landmarks) {
    auto lm = map->getLandmark(lm_id);
    int64_t frame_id = lm.host_kf_id.frame_id;
    Sophus::SE3d T_w_i = map->getKeyframePose(frame_id).cast<double>();

    const Sophus::SE3d& T_i_c = calib.T_i_c[lm.host_kf_id.cam_id];
    Mat4d T_w_c = (T_w_i * T_i_c).matrix();

    Vec4d pt_cam = StereographicParam<double>::unproject(lm.direction.cast<double>());
    pt_cam[3] = lm.inv_dist;

    Vec4d pt_w = T_w_c * pt_cam;

    landmarks_3d.emplace(lm_id, (pt_w.template head<3>() / pt_w[3]).template cast<double>());
  }

  return landmarks_3d;
}

void MapDatabase::computeMapVisualData() {
  // show landmarks
  get_map_points(map_visual_data->landmarks, map_visual_data->landmarks_ids);

  // show keyframes
  for (const auto& [frame_id, pose] : map->getKeyframes()) {
    map_visual_data->keyframe_idx[frame_id] = map->getKeyframeIndex(frame_id);
    map_visual_data->keyframe_poses[frame_id] = pose.template cast<double>();
  }

  // show covisibility
  if (config.debug1) {
    for (const auto& [tcid_h, target_map] : map->getObservations()) {
      for (const auto& [tcid_t, obs] : target_map) {
        Eigen::Vector3d p1 = map->getKeyframePose(tcid_h.frame_id).template cast<double>().translation();
        Eigen::Vector3d p2 = map->getKeyframePose(tcid_t.frame_id).template cast<double>().translation();
        map_visual_data->covisibility.emplace_back(p1);
        map_visual_data->covisibility.emplace_back(p2);
      }
    }
  }

  // Show observations
  if (config.debug1) {
    for (const auto& [tcid, obs] : map->getKeyframeObs()) {
      Eigen::Vector3d kf_pos = map->getKeyframePose(tcid.frame_id).template cast<double>().translation();
      auto landmarks_3d = get_landmarks_3d_pos(obs);
      for (const auto& lm_id : obs) {
        map_visual_data->observations[lm_id].emplace_back(kf_pos);
        map_visual_data->observations[lm_id].emplace_back(landmarks_3d[lm_id]);
      }
    }
  }
}

void MapDatabase::handleCovisibilityReq(const std::vector<size_t>& curr_kpts) {
  LandmarkDatabase<Scalar>::Ptr covisible_submap{};

  if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_DEFAULT) {
    covisible_submap = std::make_shared<LandmarkDatabase<Scalar>>("Covisible Submap");
    map->getCovisibilityMap(covisible_submap);
  } else if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_STS) {
    computeSTSMap(curr_kpts);
    covisible_submap = std::make_shared<LandmarkDatabase<Scalar>>(*sts_map);
  } else {
    BASALT_LOG_FATAL("Unexpected covisibility criteria");
  }
  if (out_covi_res_queue) out_covi_res_queue->push(covisible_submap);
}

void MapDatabase::computeSpatialDistributions(const std::set<TimeCamId>& kfs_to_compute) {
  for (const auto& [kf_id, obs] : map->getKeyframeObs()) {
    if (kfs_to_compute.count(kf_id) == 0) continue;

    auto landmarks_3d = get_landmarks_3d_pos(obs);
    std::vector<Vec3d> points;
    points.reserve(landmarks_3d.size());
    for (const auto& [lm_id, lm_3d] : landmarks_3d) points.emplace_back(lm_3d);
    keyframes_sdc[kf_id] = SpatialDistributionCube<double>(points);
  }
}

void MapDatabase::computeSTSMap(const std::vector<size_t>& curr_kpts) {
  if (map->numKeyframes() == 0) return;

  SpatialDistributionCube<double> current_sdc;

  if (config.map_sts_use_last_frame) {
    std::set<size_t> curr_lms;
    for (const auto& kpid : curr_kpts) {
      if (map->landmarkExists(kpid)) curr_lms.insert(kpid);
    }
    std::vector<Vec3d> points;
    for (const auto& [lm_id, lm_3d] : get_landmarks_3d_pos(curr_lms)) points.emplace_back(lm_3d);
    current_sdc = SpatialDistributionCube<double>(points);
  } else {
    auto last_keyframe = map->getLastKeyframe();
    current_sdc = keyframes_sdc[last_keyframe];
  }

  std::set<TimeCamId> candidate_kfs;
  for (const auto& [kf_id, sdc] : keyframes_sdc) {
    if (current_sdc.hasOverlap(keyframes_sdc[kf_id])) candidate_kfs.insert(kf_id);
    if (static_cast<int>(candidate_kfs.size()) >= config.map_sts_max_size) break;
  }

  map->getSubmap(candidate_kfs, sts_map);
}

}  // namespace basalt
