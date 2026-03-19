#include <basalt/vi_estimator/map_database.h>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace basalt {

MapDatabase::MapDatabase(const VioConfig& config, const Calibration<double>& calib, const std::string& map_export_path)
    : out_vis_queue(nullptr) {
  this->config = config;
  this->calib = calib;
  this->map = LandmarkDatabase<float>("Persistent Map");
  this->covisibility_graph = CovisibilityGraph();
  this->map_export_path = map_export_path.empty() ? "exported_map" : map_export_path;
}

void MapDatabase::write_map_stamp(basalt::MapStamp::Ptr& map_stamp) {
  if (map_stamp == nullptr) {
    return;
  }

  BASALT_ASSERT(map_stamp->lmdb->debug_check_keyframes_consistency("MapDatabase.MapStamp"));

  std::unique_lock<std::mutex> lock(mutex);

  for (const auto& [kf_id, _] : map_stamp->lmdb->getKeyframes()) {
    if (!map.keyframeExists(kf_id)) {
      active_keyframes.insert(kf_id);
    }
  }

  map.mergeLMDB(map_stamp->lmdb, true);

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

  if (pending_loop_detection != nullptr && map.keyframeExists(pending_loop_detection->current_kf_id)) {
    write_queue.push(pending_loop_detection);

    pending_loop_detection = nullptr;
  }
}

void MapDatabase::write_map_marg(std::set<FrameId>& keyframes_to_marg) {
  if (keyframes_to_marg.empty()) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex);
  for (const auto& kf_id : keyframes_to_marg) {
    active_keyframes.erase(kf_id);
  }

  if (sync_map_marg != nullptr) {
    std::lock_guard<std::mutex> lk(sync_map_marg->m);
    sync_map_marg->ready = true;
    sync_map_marg->cvar.notify_one();
  }
}

void MapDatabase::handle_loop_detection(LoopDetectionResult::Ptr& loop_detection_result) {
  std::unique_lock<std::mutex> lock(mutex);

  // If the keyframe is not in the map yet, save the loop detection for later
  if (!map.keyframeExists(loop_detection_result->current_kf_id)) {
    pending_loop_detection = loop_detection_result;
    return;
  }

  // Obtain the drift that the loop closure would correct. If it's not enough, perform the merging of landmarks,
  // but skip the pose graph optimization. If it's enough, send a map copy to the LoopClosing thread to optimize it
  Sophus::SE3f current_pose = map.getKeyframePose(loop_detection_result->current_kf_id).cast<float>();
  Sophus::SE3f corrected_pose = loop_detection_result->current_kf_corrected_pose;
  float drift_reduced = (current_pose.translation() - corrected_pose.translation()).norm();

  LoopClosureDecision::Ptr loop_closure_decision = std::make_shared<LoopClosureDecision>();
  if (drift_reduced >= config.loop_closing_min_drift_reduction) {
    auto keyframes_poses_copy = std::make_shared<Eigen::aligned_map<FrameId, Sophus::SE3f>>(map.getKeyframes());

    loop_closure_decision->keyframe_poses = keyframes_poses_copy;
    loop_closure_decision->covisibility_graph = covisibility_graph;
    loop_closure_decision->active_keyframes = active_keyframes;
    loop_closure_decision->close_loop = true;

    if (out_lc_dec_res_queue) {
      out_lc_dec_res_queue->push(loop_closure_decision);
    }

    return;
  }

  loop_closure_decision->close_loop = false;
  if (out_lc_dec_res_queue) {
    out_lc_dec_res_queue->push(loop_closure_decision);
  }

  covisibility_graph.addLoopClosure(loop_detection_result->candidate_kf_id, loop_detection_result->current_kf_id);

  merge_loop_landmarks(loop_detection_result);

  if (sync_lc_finished != nullptr) {
    std::lock_guard<std::mutex> lk(sync_lc_finished->m);
    sync_lc_finished->ready = true;
    sync_lc_finished->cvar.notify_one();
  }
}

void MapDatabase::handle_loop_closure(LoopClosureResult::Ptr& loop_closure_result) {
  std::unique_lock<std::mutex> lock(mutex);

  if (loop_closure_result->keyframe_poses != nullptr) {
    map.mergeKeyframesPoses(loop_closure_result->keyframe_poses);
  }

  LoopDetectionResult::Ptr loop_detection_result = loop_closure_result->loop_detection_result;

  covisibility_graph.addLoopClosure(loop_detection_result->candidate_kf_id, loop_detection_result->current_kf_id);

  merge_loop_landmarks(loop_detection_result);

  if (loop_closure_result->keyframe_poses != nullptr && out_opt_poses_queue && !config.causal_evaluation) {
    auto opt_poses_update = std::make_shared<basalt::OptimizedPosesUpdate>();
    opt_poses_update->keyframe_poses = loop_closure_result->keyframe_poses;
    out_opt_poses_queue->push(opt_poses_update);
  }

  if (sync_lc_finished != nullptr) {
    std::lock_guard<std::mutex> lk(sync_lc_finished->m);
    sync_lc_finished->ready = true;
    sync_lc_finished->cvar.notify_one();
  }
}

void MapDatabase::merge_loop_landmarks(const LoopDetectionResult::Ptr& loop_detection_result) {
  std::unordered_map<FrameId, std::set<LandmarkId>> covisibilities_updated;
  std::set<LandmarkId> curr_lms_to_merge;

  for (const auto& [curr_tcid, lm_obs_map] : loop_detection_result->curr_kf_obs) {
    for (const auto& [curr_lm_id, obs] : lm_obs_map) {
      LandmarkId host_lm_id = loop_detection_result->lm_fusions[curr_lm_id];

      if (!map.landmarkExists(host_lm_id)) {
        continue;
      }

      if (!map.landmarkExists(curr_lm_id) || host_lm_id == curr_lm_id) {
        Landmark<float>& host_lm = map.getLandmark(host_lm_id);

        // Increase the edge weight in the covisibility graph by 1
        if (covisibilities_updated[curr_tcid.frame_id].find(host_lm_id) ==
            covisibilities_updated[curr_tcid.frame_id].end()) {
          for (const auto& [tcid, _] : host_lm.obs) {
            if (tcid.frame_id == curr_tcid.frame_id) {
              continue;
            }
            covisibility_graph.incrementEdge(curr_tcid.frame_id, tcid.frame_id, 1);
          }
          covisibilities_updated[curr_tcid.frame_id].insert(host_lm_id);
        }

        KeypointObservation<float> kp_obs;
        kp_obs.kpt_id = host_lm_id;
        kp_obs.pos = obs;
        map.addObservation(curr_tcid, kp_obs);
      } else {
        curr_lms_to_merge.insert(curr_lm_id);
      }
    }
  }

  for (const auto& curr_lm_id : curr_lms_to_merge) {
    LandmarkId host_lm_id = loop_detection_result->lm_fusions[curr_lm_id];

    // The host landmark might have been already merged into another one
    if (!map.landmarkExists(host_lm_id)) {
      continue;
    }

    // Before merging, update the covisibility graph
    const auto& host_lm = map.getLandmark(host_lm_id);
    const auto& curr_lm = map.getLandmark(curr_lm_id);
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
        covisibility_graph.incrementEdge(old_kf_id, new_kf_id, 1);
      }
    }

    // Merge the landmarks
    map.mergeLandmarks(host_lm_id, curr_lm_id);
  }
}

void MapDatabase::read_covisibility_req(std::vector<KeypointId>& keypoints) {
  if (keypoints.empty()) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex);
  handleCovisibilityReq(keypoints);
}

void MapDatabase::read_island_req(FrameId keyframe, size_t neighbors_num) {
  IslandResponse::Ptr island_response = std::make_shared<IslandResponse>();

  island_response->keyframes.push_back(keyframe);

  std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& landmarks_3d_map =
      island_response->landmarks_3d_map;

  std::unique_lock<std::mutex> lock(mutex);

  for (const auto& covi_kf : covisibility_graph.getTopCovisible(keyframe, neighbors_num)) {
    island_response->keyframes.push_back(covi_kf);
  }

  for (const auto& kf_id : island_response->keyframes) {
    if (!map.keyframeExists(kf_id)) {
      continue;
    }

    for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
      TimeCamId kf_tcid{kf_id, cam_id};
      auto it = map.getKeyframeObs().find(kf_tcid);
      if (it == map.getKeyframeObs().end()) {
        continue;
      }
      std::set<LandmarkId> lm_ids = it->second;

      // TODO@tsantucci: this should be performed by LC
      for (const auto& lm_id : lm_ids) {
        auto lm_pos = map.getLandmark(lm_id);
        int64_t frame_id = lm_pos.host_kf_id.frame_id;
        Sophus::SE3d T_w_i = map.getKeyframePose(frame_id).cast<double>();

        const Sophus::SE3d& T_i_c = calib.T_i_c[lm_pos.host_kf_id.cam_id];
        Mat4d T_w_c = (T_w_i * T_i_c).matrix();

        Vec4d pt_cam = StereographicParam<double>::unproject(lm_pos.direction.cast<double>());
        pt_cam[3] = lm_pos.inv_dist;

        Vec4d pt_w = T_w_c * pt_cam;

        landmarks_3d_map[kf_tcid][lm_id] = (pt_w.template head<3>() / pt_w[3]).template cast<double>();
      }
    }
  }

  if (out_island_res_queue) {
    out_island_res_queue->push(island_response);
  }
}

void MapDatabase::initialize() {
  auto read_func = [&]() {
    MapReadMessage msg;
    bool running = true;
    while (running) {
      read_queue.pop(msg);

      std::visit(
          [&](auto& m) {
            using T = std::decay_t<decltype(m)>;

            if constexpr (std::is_same_v<T, CovisibilityRequest::Ptr>) {
              read_covisibility_req(m->keypoints);
            } else if constexpr (std::is_same_v<T, IslandRequest::Ptr>) {
              read_island_req(m->keyframe, m->neighbors_num);
            } else if constexpr (std::is_same_v<T, StopMsg>) {
              running = false;
            }
          },
          msg);
    }
  };

  auto write_func = [&]() {
    MapWriteMessage msg;
    bool running = true;
    while (running) {
      write_queue.pop(msg);

      std::visit(
          [&](auto& m) {
            using T = std::decay_t<decltype(m)>;

            if constexpr (std::is_same_v<T, MapStamp::Ptr>) {
              write_map_stamp(m);
            } else if constexpr (std::is_same_v<T, MarginalizationStamp::Ptr>) {
              write_map_marg(m->keyframe_ids);
            } else if constexpr (std::is_same_v<T, LoopDetectionResult::Ptr>) {
              handle_loop_detection(m);
            } else if constexpr (std::is_same_v<T, LoopClosureResult::Ptr>) {
              handle_loop_closure(m);
            } else if constexpr (std::is_same_v<T, StopMsg>) {
              map.print();
              covisibility_graph.print_stats();
              if (!map_export_path.empty()) {
                saveEuroc(map_export_path + "/tracking.csv");
                saveJson(map_export_path + "/map.json");
              }
              if (out_vis_queue) out_vis_queue->push(nullptr);
              if (out_opt_poses_queue) out_opt_poses_queue->push(nullptr);
              running = false;
            }
          },
          msg);
    }
  };

  reading_thread.reset(new std::thread(read_func));
  writing_thread.reset(new std::thread(write_func));
}

void MapDatabase::get_map_points(Eigen::aligned_vector<Vec3d>& points, std::vector<int>& ids) {
  points.clear();
  ids.clear();

  for (const auto& tcid_host : map.getHostKfs()) {
    int64_t id = tcid_host.frame_id;
    Sophus::SE3d T_w_i = map.getKeyframePose(id).cast<double>();

    const Sophus::SE3d& T_i_c = calib.T_i_c[tcid_host.cam_id];
    Mat4d T_w_c = (T_w_i * T_i_c).matrix();

    for (const auto& [lm_id, lm_pos] : map.getLandmarksForHostWithIds(tcid_host)) {
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
    auto lm = map.getLandmark(lm_id);
    int64_t frame_id = lm.host_kf_id.frame_id;
    Sophus::SE3d T_w_i = map.getKeyframePose(frame_id).cast<double>();

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
  for (const auto& [frame_id, pose] : map.getKeyframes()) {
    map_visual_data->keyframe_idx[frame_id] = map.getKeyframeIndex(frame_id);
    map_visual_data->keyframe_poses[frame_id] = pose.template cast<double>();
  }

  // show covisibility
  if (config.debug1) {
    for (const auto& [kf_id, _pose] : map.getKeyframes()) {
      if (!covisibility_graph.hasNode(kf_id)) continue;

      for (const auto& [other_kf_id, weight] : covisibility_graph.getCovisibility(kf_id)) {
        if (kf_id >= other_kf_id) continue;

        Eigen::Vector3d p1 = map.getKeyframePose(kf_id).template cast<double>().translation();
        Eigen::Vector3d p2 = map.getKeyframePose(other_kf_id).template cast<double>().translation();
        map_visual_data->covisibility.emplace_back(p1);
        map_visual_data->covisibility.emplace_back(p2);
        map_visual_data->covisibility_w.push_back(weight);
      }
    }
  }

  // show spanning tree
  for (const auto& [kf_id, _pose] : map.getKeyframes()) {
    if (!covisibility_graph.hasNode(kf_id)) continue;

    FrameId tcid_h = covisibility_graph.getParentId(kf_id);
    if (tcid_h == CovisibilityGraph::invalid()) continue;

    Eigen::Vector3d p1 = map.getKeyframePose(tcid_h).template cast<double>().translation();
    Eigen::Vector3d p2 = map.getKeyframePose(kf_id).template cast<double>().translation();
    map_visual_data->spanning_tree.emplace_back(p1);
    map_visual_data->spanning_tree.emplace_back(p2);
  }

  // show loop closures
  for (const auto& [kf_id, loop_closures] : covisibility_graph.getLoopClosures()) {
    Eigen::Vector3d p1 = map.getKeyframePose(kf_id).template cast<double>().translation();
    for (const auto& other_kf_id : loop_closures) {
      if (kf_id >= other_kf_id) continue;
      Eigen::Vector3d p2 = map.getKeyframePose(other_kf_id).template cast<double>().translation();
      map_visual_data->loop_closures.emplace_back(p1);
      map_visual_data->loop_closures.emplace_back(p2);
    }
  }

  // Show observations
  if (config.debug1) {
    for (const auto& [tcid, obs] : map.getKeyframeObs()) {
      Eigen::Vector3d kf_pos = map.getKeyframePose(tcid.frame_id).template cast<double>().translation();
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
    std::vector<FrameId> candidate_kfs =
        covisibility_graph.getCovisibleAbove(map.getLastKeyframe().frame_id, config.map_covisibility_min_weight);

    // sort the candidate_kfs from oldest to newest
    std::sort(candidate_kfs.begin(), candidate_kfs.end());

    covisible_submap = std::make_shared<LandmarkDatabase<Scalar>>("Covisible Submap");

    std::set<TimeCamId> covisible_kf_tcids;

    for (size_t i = 0; i < candidate_kfs.size() && i < static_cast<size_t>(config.map_covisibility_max_size); i++) {
      FrameId kf_id = candidate_kfs[i];
      for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
        TimeCamId tcid{kf_id, static_cast<CamId>(cam_id)};
        covisible_kf_tcids.insert(tcid);
      }
    }

    map.getSubmap(covisible_kf_tcids, covisible_submap);
  } else if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_STS) {
    computeSTSMap(curr_kpts);
    covisible_submap = std::make_shared<LandmarkDatabase<Scalar>>(*sts_map);
  } else {
    BASALT_LOG_FATAL("Unexpected covisibility criteria");
  }
  if (out_covi_res_queue) out_covi_res_queue->push(covisible_submap);
}

void MapDatabase::computeSpatialDistributions(const std::set<TimeCamId>& kfs_to_compute) {
  for (const auto& [kf_id, obs] : map.getKeyframeObs()) {
    if (kfs_to_compute.count(kf_id) == 0) continue;

    auto landmarks_3d = get_landmarks_3d_pos(obs);
    std::vector<Vec3d> points;
    points.reserve(landmarks_3d.size());
    for (const auto& [lm_id, lm_3d] : landmarks_3d) points.emplace_back(lm_3d);
    keyframes_sdc[kf_id] = SpatialDistributionCube<double>(points);
  }
}

void MapDatabase::computeSTSMap(const std::vector<size_t>& curr_kpts) {
  if (map.numKeyframes() == 0) return;

  SpatialDistributionCube<double> current_sdc;

  if (config.map_sts_use_last_frame) {
    std::set<size_t> curr_lms;
    for (const auto& kpid : curr_kpts) {
      if (map.landmarkExists(kpid)) curr_lms.insert(kpid);
    }
    std::vector<Vec3d> points;
    for (const auto& [lm_id, lm_3d] : get_landmarks_3d_pos(curr_lms)) points.emplace_back(lm_3d);
    current_sdc = SpatialDistributionCube<double>(points);
  } else {
    auto last_keyframe = map.getLastKeyframe();
    current_sdc = keyframes_sdc[last_keyframe];
  }

  std::set<TimeCamId> candidate_kfs;
  for (const auto& [kf_id, sdc] : keyframes_sdc) {
    if (current_sdc.hasOverlap(keyframes_sdc[kf_id])) candidate_kfs.insert(kf_id);
    if (static_cast<int>(candidate_kfs.size()) >= config.map_covisibility_max_size) break;
  }

  map.getSubmap(candidate_kfs, sts_map);
}

const std::map<std::string, double> MapDatabase::getStats() {
  std::map<std::string, double> stats{};
  stats["num_kfs"] = map.numKeyframes();
  stats["num_lms"] = map.numLandmarks();
  stats["num_obs"] = map.numObservations();
  return stats;
}

void MapDatabase::updateCovisibilityGraph(const LandmarkDatabase<Scalar>::Ptr& lmdb) {
  for (const auto& [kf_id, _] : lmdb->getKeyframes()) {
    if (covisibility_graph.hasNode(kf_id)) {
      continue;
    }

    if (covisibility_graph.getRoot() == CovisibilityGraph::invalid()) {
      covisibility_graph.setRoot(kf_id);
    }

    // Compute the covisibility edges
    std::unordered_map<FrameId, int> covisibilities;
    // Count each landmark only once per keyframe, even if seen from multiple cameras
    std::unordered_set<LandmarkId> counted_landmarks;
    for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
      TimeCamId kf_tcid{kf_id, static_cast<CamId>(cam_id)};

      auto obs_it = map.getKeyframeObs().find(kf_tcid);
      if (obs_it == map.getKeyframeObs().end()) {
        continue;
      }

      for (const auto& lm_id : obs_it->second) {
        // Only count each landmark once per keyframe
        if (!counted_landmarks.insert(lm_id).second) continue;

        const Landmark<float>& lm = map.getLandmark(lm_id);

        for (const auto& lm_obs : lm.obs) {
          FrameId other_kf_id = lm_obs.first.frame_id;
          if (other_kf_id == kf_id) {
            continue;
          }

          // Add covisibility edges only to existing nodes or to the root
          if (!covisibility_graph.hasNode(other_kf_id) && covisibility_graph.getRoot() != other_kf_id) {
            continue;
          }

          covisibilities[other_kf_id]++;
        }
      }
    }

    int max_weight = -1;
    FrameId max_kf_id = CovisibilityGraph::invalid();

    // Add a covisibility to the previous keyframe in the graph to avoid disconnected components
    if (covisibilities.empty() && covisibility_graph.getRoot() != kf_id) {
      auto iter = map.getKeyframes().find(kf_id);

      // Check this for safety, but it should never happen
      if (iter == map.getKeyframes().begin()) {
        continue;
      }

      --iter;
      FrameId prev_kf_id = iter->first;
      covisibilities[prev_kf_id] = 0;
    }

    for (const auto& [other_kf_id, weight] : covisibilities) {
      covisibility_graph.setEdge(kf_id, other_kf_id, weight);

      if (weight > max_weight) {
        max_weight = weight;
        max_kf_id = other_kf_id;
      }
    }

    // Finally add it to the spanning tree
    covisibility_graph.addTreeNode(kf_id, max_kf_id);
  }
}

void MapDatabase::saveEuroc(const std::string& file_path) {
  std::unique_lock<std::mutex> lock(mutex);

  std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());

  std::ofstream os(file_path);

  os << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
        "[],q_RS_x [],q_RS_y [],q_RS_z []"
     << std::endl;

  for (const auto& [kf_id, pose] : map.getKeyframes()) {
    os << std::fixed << std::setprecision(10) << kf_id << "," << pose.translation().x() << "," << pose.translation().y()
       << "," << pose.translation().z() << "," << pose.unit_quaternion().w() << "," << pose.unit_quaternion().x() << ","
       << pose.unit_quaternion().y() << "," << pose.unit_quaternion().z() << std::endl;
  }

  std::cout << "Saved trajectory in Euroc Dataset format in " << file_path << std::endl;
}

void MapDatabase::saveJson(const std::string& file_path) {
  std::unique_lock<std::mutex> lock(mutex);

  std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());
  std::ofstream json_file;
  json_file.open(file_path);

  nlohmann::json j;

  // Identify valid landmarks (at least 2 observations and finite position)
  std::unordered_set<LandmarkId> valid_lm_ids;
  for (const auto& [lm_id, lm] : map.getLandmarks()) {
    if (lm.obs.size() < 2) {
      continue;
    }

    auto landmarks_3d = get_landmarks_3d_pos({lm_id});
    Vec3d lm_3d = landmarks_3d.at(lm_id);

    if (!std::isfinite(lm_3d.x()) || !std::isfinite(lm_3d.y()) || !std::isfinite(lm_3d.z())) {
      continue;
    }

    valid_lm_ids.insert(lm_id);
  }

  // Save keyframes
  nlohmann::json keyframes_json = nlohmann::json::array();
  for (const auto& [kf_id, pose] : map.getKeyframes()) {
    Sophus::SE3d T_w_i = pose.template cast<double>();
    Eigen::Quaterniond q = T_w_i.unit_quaternion();
    Eigen::Vector3d t = T_w_i.translation();

    nlohmann::json kf_json;
    kf_json["id"] = kf_id;
    kf_json["T_w_i"] = {q.w(), q.x(), q.y(), q.z(), t.x(), t.y(), t.z()};
    keyframes_json.push_back(kf_json);
  }
  j["keyframes"] = keyframes_json;

  // Save landmarks
  nlohmann::json landmarks_json = nlohmann::json::array();
  for (const auto& lm_id : valid_lm_ids) {
    auto landmarks_3d = get_landmarks_3d_pos({lm_id});
    Vec3d lm_3d = landmarks_3d.at(lm_id);

    nlohmann::json lm_json;
    lm_json["id"] = lm_id;
    lm_json["p_w"] = {lm_3d.x(), lm_3d.y(), lm_3d.z()};
    landmarks_json.push_back(lm_json);
  }
  j["landmarks"] = landmarks_json;

  // Save observations
  nlohmann::json observations_json = nlohmann::json::array();
  for (const auto& [tcid, obs_set] : map.getKeyframeObs()) {
    for (const auto& lm_id : obs_set) {
      if (valid_lm_ids.find(lm_id) == valid_lm_ids.end()) {
        continue;
      }

      const auto& lm = map.getLandmark(lm_id);
      const auto& pt = lm.obs.at(tcid);

      nlohmann::json obs_json;
      obs_json["kf_id"] = tcid.frame_id;
      obs_json["cam_id"] = tcid.cam_id;
      obs_json["lm_id"] = lm_id;
      obs_json["pos"] = {pt.x(), pt.y()};
      observations_json.push_back(obs_json);
    }
  }
  j["observations"] = observations_json;

  json_file << j.dump(2);
  json_file.close();
}

void MapDatabase::saveColmap(const std::string& path) {
  std::unique_lock<std::mutex> lock(mutex);

  // create the cameras.txt file
  std::ofstream colmap_cameras_txt;
  std::filesystem::create_directories(path);
  colmap_cameras_txt.open(path + "/cameras.txt");
  colmap_cameras_txt << "# Camera list with one line of data per camera:\n";
  colmap_cameras_txt << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
  colmap_cameras_txt << "# Number of cameras: " << calib.intrinsics.size() << "\n";

  std::ofstream colmap_images_txt;
  colmap_images_txt.open(path + "/images.txt");
  colmap_images_txt << "# Image list with two lines of data per image:\n";
  colmap_images_txt << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
  colmap_images_txt << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";

  std::ofstream colmap_points3D_txt;
  colmap_points3D_txt.open(path + "/points3D.txt");
  colmap_points3D_txt << "# 3D point list with one line of data per point:\n";
  colmap_points3D_txt << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";

  // Also create a file that maps the sequential colmap image ids to the original frame ids for easier debugging
  std::ofstream colmap_image_id_to_frame_id_txt;
  colmap_image_id_to_frame_id_txt.open(path + "/image_id_to_frame_id.txt");

  // Write the cameras.txt file
  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    GenericCamera<double>& cam = calib.intrinsics[i];
    int w = calib.resolution[i].x();
    int h = calib.resolution[i].y();
    std::string model = cam.getName();
    Eigen::VectorXd params = cam.getParam();

    std::string colmap_model{""};  // https://colmap.github.io/cameras.html
    if (model == "pinhole")
      colmap_model = "PINHOLE";
    else if (model == "pinhole-radtan8")
      colmap_model = "FULL_OPENCV";
    else if (model == "kb4")
      colmap_model = "OPENCV_FISHEYE";
    else
      BASALT_ASSERT_MSG(false, "Camera model unavailable in colmap");

    colmap_cameras_txt << i + 1 << " ";
    colmap_cameras_txt << colmap_model << " ";
    colmap_cameras_txt << w << " ";
    colmap_cameras_txt << h;
    for (double p : params) colmap_cameras_txt << " " << p;
    colmap_cameras_txt << "\n";
  }
  colmap_cameras_txt.close();

  // Write the images.txt
  int next_image_id = 1;
  int next_point_id = 1;
  std::unordered_map<LandmarkId, int> lm_to_colmap_id;
  std::unordered_map<LandmarkId, std::vector<std::pair<int, int>>> lm_tracks;

  // As only the landmarks observed by at least 2 keyframes are considered valid for colmap, we first collect the valid
  // landmark ids.
  std::unordered_set<LandmarkId> valid_lm_ids;
  for (const auto& [lm_id, lm] : map.getLandmarks()) {
    int obs_count = 0;
    bool valid = true;
    for (const auto& [tcid, _] : lm.obs) {
      // if the tcid.cam_id is not the the cameras to export vector, then the landmark is not valid for colmap
      if (std::find(config.cameras_to_export.begin(), config.cameras_to_export.end(), tcid.cam_id) ==
          config.cameras_to_export.end()) {
        continue;
      }
      obs_count++;
    }
    if (obs_count >= 2) {
      valid_lm_ids.insert(lm_id);
    }
  }

  for (const auto& cam_id : config.cameras_to_export) {
    Sophus::SE3d T_i_c = calib.T_i_c[cam_id];
    for (const auto& [kf_id, pose] : map.getKeyframes()) {
      TimeCamId kf_tcid{kf_id, static_cast<CamId>(cam_id)};
      auto obs_it = map.getKeyframeObs().find(kf_tcid);
      if (obs_it == map.getKeyframeObs().end()) {
        continue;
      }

      Sophus::SE3d T_w_c = pose.template cast<double>() * T_i_c;
      Sophus::SE3d T_c_w = T_w_c.inverse();
      Eigen::Quaterniond q = T_c_w.unit_quaternion();
      Eigen::Vector3d t = T_c_w.translation();

      colmap_images_txt << next_image_id << " ";
      colmap_images_txt << q.w() << " ";
      colmap_images_txt << q.x() << " ";
      colmap_images_txt << q.y() << " ";
      colmap_images_txt << q.z() << " ";
      colmap_images_txt << t.x() << " ";
      colmap_images_txt << t.y() << " ";
      colmap_images_txt << t.z() << " ";
      colmap_images_txt << cam_id + 1 << " ";
      colmap_images_txt << frame_id_to_name->at(kf_tcid);
      colmap_images_txt << "\n";

      colmap_image_id_to_frame_id_txt << next_image_id << " " << kf_id << " " << cam_id + 1 << "\n";

      int point2d_idx = 0;

      for (const auto& lm_id : obs_it->second) {
        if (valid_lm_ids.find(lm_id) == valid_lm_ids.end()) {
          continue;
        }

        if (lm_to_colmap_id.find(lm_id) == lm_to_colmap_id.end()) {
          lm_to_colmap_id[lm_id] = next_point_id;
          next_point_id++;
        }

        auto landmarks_3d = get_landmarks_3d_pos({lm_id});
        Vec3d lm_3d = landmarks_3d.at(lm_id);

        // Skip the inf points
        if (!std::isfinite(lm_3d.x()) || !std::isfinite(lm_3d.y()) || !std::isfinite(lm_3d.z())) {
          continue;
        }

        int pid = lm_to_colmap_id[lm_id];
        const auto& lm = map.getLandmark(lm_id);
        const auto& pt = lm.obs.at(kf_tcid);

        colmap_images_txt << pt.x() << " ";
        colmap_images_txt << pt.y() << " ";
        colmap_images_txt << pid << " ";

        lm_tracks[lm_id].emplace_back(next_image_id, point2d_idx);

        point2d_idx++;
      }
      colmap_images_txt << "\n";

      next_image_id++;
    }
  }
  colmap_images_txt.close();
  colmap_image_id_to_frame_id_txt.close();

  // Write the points3D.txt
  for (const auto& [lm_id, lm] : map.getLandmarks()) {
    if (valid_lm_ids.find(lm_id) == valid_lm_ids.end()) {
      continue;
    }

    if (lm_to_colmap_id.find(lm_id) == lm_to_colmap_id.end()) {
      continue;
    }

    auto landmarks_3d = get_landmarks_3d_pos({lm_id});
    Vec3d lm_3d = landmarks_3d.at(lm_id);

    // Skip the inf points
    if (!std::isfinite(lm_3d.x()) || !std::isfinite(lm_3d.y()) || !std::isfinite(lm_3d.z())) {
      continue;
    }

    colmap_points3D_txt << lm_to_colmap_id[lm_id] << " ";
    colmap_points3D_txt << lm_3d.x() << " ";
    colmap_points3D_txt << lm_3d.y() << " ";
    colmap_points3D_txt << lm_3d.z() << " ";
    colmap_points3D_txt << "0 0 0 0.0";  // Using a fixed color and error for simplicity

    const auto& tracks = lm_tracks[lm_id];
    for (const auto& [img_id, p2d_idx] : tracks) {
      colmap_points3D_txt << " " << img_id;
      colmap_points3D_txt << " " << p2d_idx;
    }
    colmap_points3D_txt << "\n";
  }
  colmap_points3D_txt.close();
}

}  // namespace basalt
