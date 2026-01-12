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
  this->covisibility_graph = std::make_shared<CovisibilityGraph>();
}

void MapDatabase::write_map_stamp(basalt::MapStamp::Ptr& map_stamp) {
  if (map_stamp == nullptr) {
    map->print();
    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_map_update_queue) out_map_update_queue->push(nullptr);
    return;
  }

  BASALT_ASSERT(map_stamp->lmdb->debug_check_keyframes_consistency("MapDatabase.MapStamp"));

  std::unique_lock<std::mutex> lock(mutex);
  map->mergeLMDB(map_stamp->lmdb, true);

  if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_STS) {
    std::set<TimeCamId> kfs_to_compute;
    for (auto const& [kf_id, _] : map_stamp->lmdb->getKeyframeObs()) kfs_to_compute.emplace(kf_id);
    computeSpatialDistributions(kfs_to_compute);
  }

  // Add the new keyframes to the covisibility graph
  updateCovisibilityGraph(map_stamp->lmdb);

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

void MapDatabase::write_map_update(
    std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>>& keyframe_poses, FrameId candidate_kf_id,
    FrameId curr_kf_id, std::unordered_map<LandmarkId, LandmarkId>& lm_fusions,
    std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Eigen::Matrix<float, 2, 1>>>& curr_lc_obs) {
  if (keyframe_poses == nullptr) return;

  std::unique_lock<std::mutex> lock(mutex);

  map->mergeKeyframesPoses(keyframe_poses);

  covisibility_graph->addLoopClosure(candidate_kf_id, curr_kf_id);

  std::unordered_map<FrameId, std::set<LandmarkId>> covisibilities_updated;
  std::set<LandmarkId> curr_lms_to_merge;

  // Perform loop fusion
  for (const auto& [curr_tcid, lm_obs_map] : curr_lc_obs) {
    for (const auto& [curr_lm_id, obs] : lm_obs_map) {
      LandmarkId host_lm_id = lm_fusions[curr_lm_id];

      BASALT_ASSERT(map->landmarkExists(host_lm_id));

      if (!map->landmarkExists(curr_lm_id) || host_lm_id == curr_lm_id) {
        Landmark<float>& host_lm = map->getLandmark(host_lm_id);

        // should increase the edge weight in the covisibility graph by 1
        if (covisibilities_updated[curr_tcid.frame_id].find(host_lm_id) ==
            covisibilities_updated[curr_tcid.frame_id].end()) {
          for (const auto& [tcid, _] : host_lm.obs) {
            if (tcid.frame_id == curr_tcid.frame_id) {
              continue;
            }
            covisibility_graph->updateEdge(curr_tcid.frame_id, tcid.frame_id, 1);
          }
          covisibilities_updated[curr_tcid.frame_id].insert(host_lm_id);
        }

        KeypointObservation<float> kp_obs;
        kp_obs.kpt_id = host_lm_id;
        kp_obs.pos = obs;
        map->addObservation(curr_tcid, kp_obs);
      } else {
        curr_lms_to_merge.insert(curr_lm_id);
      }
    }
  }

  for (const auto& curr_lm_id : curr_lms_to_merge) {
    LandmarkId host_lm_id = lm_fusions[curr_lm_id];

    // The host landmark might have been already merged into another one
    if (!map->landmarkExists(host_lm_id)) {
      continue;
    }

    // Before merging, update the covisibility graph
    const auto& host_lm = map->getLandmark(host_lm_id);
    const auto& curr_lm = map->getLandmark(curr_lm_id);
    std::set<FrameId> old_kf_obs;
    std::set<FrameId> new_kf_obs;
    for (const auto& [tcid_host, _] : host_lm.obs) {
      old_kf_obs.insert(tcid_host.frame_id);
    }
    for (const auto& [curr_tcid, _] : curr_lm.obs) {
      new_kf_obs.insert(curr_tcid.frame_id);
    }

    for (const auto& old_kf_id : old_kf_obs) {
      for (const auto& new_kf_id : new_kf_obs) {
        if (old_kf_id == new_kf_id) {
          continue;
        }
        covisibility_graph->updateEdge(old_kf_id, new_kf_id, 1);
      }
    }

    // Merge the landmarks
    map->mergeLandmarks(host_lm_id, curr_lm_id);
  }

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

void MapDatabase::read_3d_points_req(FrameId keyframe, size_t neighbors_num) {
  std::unique_lock<std::mutex> lock(mutex);

  std::vector<FrameId> keyframes = {keyframe};

  for (const auto& covi_kf : covisibility_graph->getTopK(keyframe, neighbors_num)) {
    keyframes.push_back(covi_kf);
  }

  auto landmarks_3d_map = std::make_shared<std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>>();

  for (const auto& kf_id : keyframes) {
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
    MapIslandResponse::Ptr map_island_response = std::make_shared<MapIslandResponse>();
    map_island_response->island_keyframes = keyframes;
    map_island_response->landmarks_3d_map = *landmarks_3d_map;
    out_3d_points_res_queue->push(map_island_response);
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

  auto covisibility_graph_copy = std::make_shared<CovisibilityGraph>(*covisibility_graph);

  MapResponse::Ptr map_response = std::make_shared<MapResponse>();
  map_response->keyframe_poses = keyframes_poses_copy;
  map_response->covisibility_graph = covisibility_graph_copy;

  if (out_map_res_queue) {
    out_map_res_queue->push(map_response);
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
    for (const auto& [kf_id, _pose] : map->getKeyframes()) {
      if (!covisibility_graph->hasNode(kf_id)) continue;

      for (const auto& [other_kf_id, weight] : covisibility_graph->getCovisibleKfs(kf_id)) {
        Eigen::Vector3d p1 = map->getKeyframePose(kf_id).template cast<double>().translation();
        Eigen::Vector3d p2 = map->getKeyframePose(other_kf_id).template cast<double>().translation();
        map_visual_data->covisibility.emplace_back(p1);
        map_visual_data->covisibility.emplace_back(p2);
        map_visual_data->covisibility_w.push_back(weight);
      }
    }
  }

  // show spanning tree
  if (config.debug1) {
    for (const auto& [kf_id, _pose] : map->getKeyframes()) {
      if (!covisibility_graph->hasNode(kf_id)) continue;

      FrameId tcid_h = covisibility_graph->getParentNode(kf_id);
      if (tcid_h == CovisibilityGraph::invalid()) continue;

      Eigen::Vector3d p1 = map->getKeyframePose(tcid_h).template cast<double>().translation();
      Eigen::Vector3d p2 = map->getKeyframePose(kf_id).template cast<double>().translation();
      map_visual_data->spanning_tree.emplace_back(p1);
      map_visual_data->spanning_tree.emplace_back(p2);
    }
  }

  // show loop closures
  if (config.debug1) {
    for (const auto& [kf_id, loop_closures] : covisibility_graph->getAllLoopClosures()) {
      Eigen::Vector3d p1 = map->getKeyframePose(kf_id).template cast<double>().translation();
      for (const auto& other_kf_id : loop_closures) {
        if (kf_id >= other_kf_id) continue;
        Eigen::Vector3d p2 = map->getKeyframePose(other_kf_id).template cast<double>().translation();
        map_visual_data->loop_closures.emplace_back(p1);
        map_visual_data->loop_closures.emplace_back(p2);
      }
    }
  }

  // Show observations
  if (config.debug2) {
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

void MapDatabase::updateCovisibilityGraph(const LandmarkDatabase<Scalar>::Ptr& lmdb) {
  // TODO@tsantucci: generalize this for when the covisibility request is on.
  // It may be interesting to use all the keyframes in the lmdb, not only the new ones.
  for (const auto& [kf_id, _] : lmdb->getKeyframes()) {
    if (covisibility_graph->hasNode(kf_id)) {
      continue;
    }

    if (covisibility_graph->getRoot() == CovisibilityGraph::invalid()) {
      covisibility_graph->setRoot(kf_id);
    }

    // Compute the covisibility edges
    // Count each landmark only once per keyframe, even if seen from multiple cameras
    std::unordered_map<FrameId, int> covisibilities;
    std::unordered_set<LandmarkId> counted_landmarks;
    for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
      TimeCamId kf_tcid{kf_id, static_cast<CamId>(cam_id)};

      auto obs_it = map->getKeyframeObs().find(kf_tcid);
      if (obs_it == map->getKeyframeObs().end()) {
        continue;
      }

      for (const auto& lm_id : obs_it->second) {
        // Only count each landmark once per keyframe
        if (!counted_landmarks.insert(lm_id).second) continue;

        const Landmark<float>& lm = map->getLandmark(lm_id);

        for (const auto& lm_obs : lm.obs) {
          FrameId other_kf_id = lm_obs.first.frame_id;
          if (other_kf_id == kf_id) {
            continue;
          }

          // Add covisibility edges only to existing nodes or to the root
          if (!covisibility_graph->hasNode(other_kf_id) && covisibility_graph->getRoot() != other_kf_id) {
            continue;
          }

          covisibilities[other_kf_id]++;
        }
      }
    }

    int max_weight = -1;
    FrameId max_kf_id = CovisibilityGraph::invalid();

    // Add a covisibility to the previous keyframe in the graph to avoid disconnected components
    if (covisibilities.empty() && covisibility_graph->getRoot() != kf_id) {
      auto iter = map->getKeyframes().find(kf_id);
      if (iter == map->getKeyframes().begin()) continue;  // this should never happen anyway
      --iter;
      FrameId prev_kf_id = iter->first;
      covisibilities[prev_kf_id] = 0;
    }

    for (const auto& [other_kf_id, weight] : covisibilities) {
      covisibility_graph->addEdge(kf_id, other_kf_id, weight);

      if (weight > max_weight) {
        max_weight = weight;
        max_kf_id = other_kf_id;
      }
    }

    // add to spanning tree
    covisibility_graph->addTreeNode(kf_id, max_kf_id);
  }
}

}  // namespace basalt
