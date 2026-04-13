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
  this->covisibility_graph.setHighCovisibilityThreshold(config.loop_closing_pgo_min_covisibility_weight);
  this->covisibility_graph.setGraphScoreChangedCallback(
      [this](FrameId kf_id, NodeScore new_score) { handleGraphScoreChange(kf_id, new_score); });
  this->map_export_path = map_export_path.empty() ? "exported_map" : map_export_path;
}

void MapDatabase::write_map_stamp(basalt::MapStamp::Ptr& map_stamp) {
  if (map_stamp == nullptr) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex);

  for (const auto& [kf_id, _] : map_stamp->lmdb->getKeyframes()) {
    if (!map.keyframeExists(kf_id)) {
      active_keyframes.insert(kf_id);
    }
  }

  map_stamp->lmdb->removeReferencesToCulledKeyframes(map, most_recent_ts);

  map.mergeLMDB(map_stamp->lmdb, true);

  FrameId curr_most_recent_ts = map.getKeyframes().empty() ? 0 : map.getKeyframes().rbegin()->first;
  if (curr_most_recent_ts > most_recent_ts) {
    most_recent_ts = curr_most_recent_ts;
  }

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

  if (map.getKeyframes().size() > static_cast<size_t>(config.map_max_keyframes)) {
    if (config.map_debug)
      std::cout << "[MDB] Number of keyframes in the map: " << map.getKeyframes().size()
                << ". Starting culling process." << std::endl;
    cullKeyframes(config.map_batch_cull_size);
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

  std::vector<FrameId> culled_keyframes;
  for (const auto& kf_id : keyframes_to_marg) {
    active_keyframes.erase(kf_id);

    if (config.map_enable_culling) {
      float redundancy =
          map.getKeyframeRedundancyScore(kf_id, calib.intrinsics.size(), config.map_redundancy_exclude_hosted_lms);

      if (redundancy >= config.map_max_redundancy_score) {
        bool success = cullKeyframe(kf_id);
        if (!success) continue;

        if (config.map_debug)
          std::cout << "[MDB] Culled keyframe " << kf_id << " with redundancy score " << redundancy << std::endl;
        culled_keyframes.push_back(kf_id);
      } else {
        KeyframeScore& score = keyframe_scores[kf_id];
        score.img_bbox_coverage = computeImgCoverageScore(kf_id);
        score.total_observed_lms_score = computeTotalObservedLmsScore(kf_id);
      }
    }
  }

  if (out_culled_map_data_queue && !culled_keyframes.empty()) {
    auto culled_map_data = std::make_shared<CulledMapData>();
    culled_map_data->culled_keyframes = culled_keyframes;
    out_culled_map_data_queue->push(culled_map_data);
  }

  if (sync_map_marg != nullptr) {
    std::lock_guard<std::mutex> lk(sync_map_marg->m);
    sync_map_marg->ready = true;
    sync_map_marg->cvar.notify_one();
  }
}

bool MapDatabase::cullKeyframe(FrameId kf_id) {
  if (kf_id == covisibility_graph.getRoot()) {
    if (config.map_debug)
      std::cout << "[MDB] Skipping culling of keyframe " << kf_id << " because it is the root of the covisibility graph"
                << std::endl;
    return false;
  }

  if (current_loop_keyframes.count(kf_id) > 0) {
    if (config.map_debug)
      std::cout << "[MDB] Skipping culling of keyframe " << kf_id
                << " because it is part of the current loop closure candidate keyframes" << std::endl;
    return false;
  }

  std::vector<Landmark<float>> removed_landmarks;

  map.removeKeyframe(kf_id, calib.intrinsics.size(), removed_landmarks);

  covisibility_graph.removeNode(kf_id);

  keyframe_scores.erase(kf_id);

  for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
    TimeCamId tcid{kf_id, cam_id};
    keyframes_sdc.erase(tcid);
  }

  // update the edge weights of the covisibility graph
  for (const auto& lm : removed_landmarks) {
    std::unordered_set<FrameId> observing_kfs;
    for (const auto& [tcid, _] : lm.obs) {
      FrameId other_kf_id = tcid.frame_id;
      if (other_kf_id == kf_id) continue;

      observing_kfs.insert(other_kf_id);
    }

    for (const auto& kf1 : observing_kfs) {
      for (const auto& kf2 : observing_kfs) {
        if (kf1 >= kf2) continue;

        covisibility_graph.decrementEdge(kf1, kf2, 1);
      }
    }
  }

  return true;
}

void MapDatabase::handle_loop_detection(LoopDetectionResult::Ptr& loop_detection_result) {
  std::unique_lock<std::mutex> lock(mutex);

  // If the current keyframe of the loop detection result doesnt exist in the map, there are two possibilities:
  // 1) The keyframe has not been added to the map yet, in which case we store the loop detection result as pending
  // 2) The keyframe has been culled from the map, in which case we discard the loop detection result
  if (!map.keyframeExists(loop_detection_result->current_kf_id)) {
    FrameId newest_kf_id = map.getKeyframes().empty() ? 0 : map.getKeyframes().rbegin()->first;
    if (newest_kf_id < loop_detection_result->current_kf_id) {
      pending_loop_detection = loop_detection_result;
    } else {
      LoopClosureDecision::Ptr loop_closure_decision = std::make_shared<LoopClosureDecision>();
      loop_closure_decision->close_loop = false;
      if (out_lc_dec_res_queue) {
        out_lc_dec_res_queue->push(loop_closure_decision);
      }
    }

    return;
  }

  // if the candidate keyframe doesnt exist in the map, just discard the loop detection
  if (!map.keyframeExists(loop_detection_result->candidate_kf_id)) {
    LoopClosureDecision::Ptr loop_closure_decision = std::make_shared<LoopClosureDecision>();
    loop_closure_decision->close_loop = false;
    if (out_lc_dec_res_queue) {
      out_lc_dec_res_queue->push(loop_closure_decision);
    }
    return;
  }

  // Obtain the drift that the loop closure would correct. If it's not enough, perform the merging of landmarks,
  // but skip the pose graph optimization. If it's enough, send a map copy to the LoopClosing thread to optimize it
  Sophus::SE3f current_pose = map.getKeyframePose(loop_detection_result->current_kf_id).cast<float>();
  Sophus::SE3f corrected_pose = loop_detection_result->current_kf_corrected_pose;
  float drift_reduced = (current_pose.translation() - corrected_pose.translation()).norm();
  Sophus::SO3f rot_diff = current_pose.so3().inverse() * corrected_pose.so3();
  float rot_angle_deg = rot_diff.log().norm() * (180.0f / M_PI);

  if (config.loop_closing_debug) {
    bool accepted = drift_reduced >= config.loop_closing_min_drift_reduction ||
                    rot_angle_deg >= config.loop_closing_min_rot_reduction_deg;
    std::cout << "[MDB] Drift: " << drift_reduced << " m, rot: " << rot_angle_deg << " deg. -> "
              << (accepted ? "ACCEPTED" : "REJECTED") << std::endl;
  }

  LoopClosureDecision::Ptr loop_closure_decision = std::make_shared<LoopClosureDecision>();
  if (drift_reduced >= config.loop_closing_min_drift_reduction ||
      rot_angle_deg >= config.loop_closing_min_rot_reduction_deg) {
    auto keyframes_poses_copy = std::make_shared<Eigen::aligned_map<FrameId, Sophus::SE3f>>(map.getKeyframes());

    loop_closure_decision->keyframe_poses = keyframes_poses_copy;
    loop_closure_decision->covisibility_graph = covisibility_graph.copyPoseGraph();
    loop_closure_decision->active_keyframes = active_keyframes;
    loop_closure_decision->close_loop = true;

    current_loop_keyframes.insert(loop_detection_result->current_kf_id);
    current_loop_keyframes.insert(loop_detection_result->candidate_kf_id);

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

  current_loop_keyframes.erase(loop_detection_result->current_kf_id);
  current_loop_keyframes.erase(loop_detection_result->candidate_kf_id);

  if (sync_lc_finished != nullptr) {
    std::lock_guard<std::mutex> lk(sync_lc_finished->m);
    sync_lc_finished->ready = true;
    sync_lc_finished->cvar.notify_one();
  }
}

void MapDatabase::merge_loop_landmarks(const LoopDetectionResult::Ptr& loop_detection_result) {
  std::unordered_map<FrameId, std::unordered_set<LandmarkId>> covisibilities_updated;
  std::unordered_set<LandmarkId> curr_lms_to_merge;
  size_t num_observations_added = 0;
  size_t num_merged_lms = 0;
  std::unordered_map<CovisibilityEdge, int, CovisibilityEdge::Hash> covisibility_increments;

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
            covisibility_increments[CovisibilityEdge(curr_tcid.frame_id, tcid.frame_id)]++;
          }
          covisibilities_updated[curr_tcid.frame_id].insert(host_lm_id);
        }

        KeypointObservation<float> kp_obs;
        kp_obs.kpt_id = host_lm_id;
        kp_obs.pos = obs.translation();
        map.addObservation(curr_tcid, kp_obs);
        num_observations_added++;
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
        covisibility_increments[CovisibilityEdge(old_kf_id, new_kf_id)]++;
      }
    }

    // Merge the landmarks
    map.mergeLandmarks(host_lm_id, curr_lm_id);
    num_merged_lms++;
  }

  for (const auto& [covi_edge, increment] : covisibility_increments) {
    covisibility_graph.incrementEdge(covi_edge.id1, covi_edge.id2, increment);
  }

  if (config.loop_closing_debug) {
    std::cout << "[MDB] Merged " << num_merged_lms << " landmarks, " << num_observations_added << " observations added."
              << std::endl;
  }
}

void MapDatabase::read_covisibility_req(std::vector<KeypointId>& keypoints) {
  if (keypoints.empty()) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex);
  handleCovisibilityReq(keypoints);
}

void MapDatabase::read_island_req(FrameId keyframe, size_t num_neighbors) {
  IslandResponse::Ptr island_response = std::make_shared<IslandResponse>();

  std::unique_lock<std::mutex> lock(mutex);

  if (!map.keyframeExists(keyframe)) {
    if (out_island_res_queue) {
      out_island_res_queue->push(island_response);
    }
    return;
  }

  island_response->keyframes.push_back(keyframe);

  for (const auto& covi_kf : covisibility_graph.getTopCovisible(keyframe, num_neighbors)) {
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
      island_response->keyframe_obs[kf_tcid] = lm_ids;

      for (const auto& lm_id : lm_ids) {
        if (island_response->landmarks_3d.find(lm_id) != island_response->landmarks_3d.end()) {
          continue;
        }
        auto lm_pos = map.getLandmark(lm_id);
        int64_t frame_id = lm_pos.host_kf_id.frame_id;
        Sophus::SE3d T_w_i = map.getKeyframePose(frame_id).cast<double>();

        const Sophus::SE3d& T_i_c = calib.T_i_c[lm_pos.host_kf_id.cam_id];
        Mat4d T_w_c = (T_w_i * T_i_c).matrix();

        Vec4d pt_cam = StereographicParam<double>::unproject(lm_pos.direction.cast<double>());
        pt_cam[3] = lm_pos.inv_dist;

        Vec4d pt_w = T_w_c * pt_cam;

        island_response->landmarks_3d[lm_id] = (pt_w.template head<3>() / pt_w[3]).template cast<double>();
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
              read_island_req(m->keyframe, m->num_neighbors);
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

void MapDatabase::cullKeyframes(int num_to_cull) {
  std::vector<FrameId> culled_keyframes;
  std::vector<KeyframeScore> culled_keyframe_scores;

  // Sort all the keyframes by their score using a set
  keyframe_score_set.clear();
  const auto& all_kfs = map.getKeyframes();
  FrameId t_min = all_kfs.empty() ? 0 : all_kfs.begin()->first;
  FrameId t_max = all_kfs.empty() ? 0 : all_kfs.rbegin()->first;
  double t_range = static_cast<double>(t_max - t_min);
  for (auto& [kf_id, kf_score] : keyframe_scores) {
    kf_score.time_score = computeTimeScore(kf_id, t_min, t_range);
    keyframe_score_set.insert({computeKeyframeScore(kf_score), kf_id});
  }

  while (num_to_cull > 0) {
    // Take the keyframe with the lowest score
    auto it = keyframe_score_set.begin();
    if (it == keyframe_score_set.end()) {
      break;
    }

    FrameId kf_id = it->second;

    // If the keyframe is still active, skip it
    if (active_keyframes.count(kf_id)) {
      keyframe_score_set.erase(it);
      continue;
    }

    KeyframeScore saved_score = keyframe_scores.count(kf_id) ? keyframe_scores.at(kf_id) : KeyframeScore{};
    if (cullKeyframe(kf_id)) {
      culled_keyframes.push_back(kf_id);
      culled_keyframe_scores.push_back(saved_score);
      keyframe_score_set.erase(it);
      num_to_cull--;
    } else {
      // If the keyframe couldn't be culled, remove it from the keyframe_score_set to avoid trying to cull it again in
      // this round
      keyframe_score_set.erase(it);
    }
  }

  if (config.map_debug) {
    std::cout << "[MDB] Culled " << culled_keyframes.size() << " keyframes:" << std::endl;
    for (size_t i = 0; i < culled_keyframes.size(); ++i) {
      std::cout << culled_keyframes[i] << " - score: " << culled_keyframe_scores[i].node_score.graph_score
                << " (graph), " << culled_keyframe_scores[i].time_score << " (time), "
                << culled_keyframe_scores[i].img_bbox_coverage << " (img bbox coverage), "
                << culled_keyframe_scores[i].total_observed_lms_score << " (observed lms), "
                << culled_keyframe_scores[i].node_score.loop_score << " (loop score)" << std::endl;
    }
    std::cout << std::endl;
  }

  if (out_culled_map_data_queue && !culled_keyframes.empty()) {
    auto culled_map_data = std::make_shared<CulledMapData>();
    culled_map_data->culled_keyframes = culled_keyframes;
    out_culled_map_data_queue->push(culled_map_data);
  }

  keyframe_score_set.clear();
}

void MapDatabase::handleGraphScoreChange(FrameId kf_id, NodeScore new_score) {
  // If the keyframe doesn't have a score entry yet, default-construct it
  KeyframeScore& entry = keyframe_scores[kf_id];

  KeyframeScore prev_score = entry;
  entry.node_score = new_score;

  if (keyframe_score_set.size() > 0) {
    // Update the score of the keyframe in the keyframe_score_set
    auto set_it = keyframe_score_set.find({computeKeyframeScore(prev_score), kf_id});
    if (set_it != keyframe_score_set.end()) {
      keyframe_score_set.erase(set_it);
      keyframe_score_set.insert({computeKeyframeScore(entry), kf_id});
    }
  }
}

float MapDatabase::computeImgCoverageScore(FrameId kf_id) {
  float img_area = 0.0f;
  float bbox_area = 0.0f;
  std::vector<float> ratios;

  for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
    TimeCamId tcid{kf_id, cam_id};

    if (map.getKeyframeObs().count(tcid) == 0) continue;

    const auto& kpts = map.getKeyframeObs().at(tcid);
    if (kpts.size() > 0) {
      const auto& resolution = calib.resolution[cam_id];
      img_area += resolution.x() * resolution.y();

      float min_x = std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float max_x = std::numeric_limits<float>::lowest();
      float max_y = std::numeric_limits<float>::lowest();

      for (const auto& kp : kpts) {
        auto pos = map.getLandmark(kp).obs.at(tcid);
        min_x = std::min(min_x, pos.x());
        min_y = std::min(min_y, pos.y());
        max_x = std::max(max_x, pos.x());
        max_y = std::max(max_y, pos.y());
      }

      bbox_area += (max_x - min_x) * (max_y - min_y);
      ratios.push_back(bbox_area / img_area);
    }
  }

  float img_bbox_coverage =
      (ratios.empty()) ? 0.0f : std::accumulate(ratios.begin(), ratios.end(), 0.0f) / ratios.size();

  return img_bbox_coverage;
}

float MapDatabase::computeTotalObservedLmsScore(FrameId kf_id) {
  std::unordered_set<LandmarkId> unique_lms;
  for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
    TimeCamId tcid{kf_id, cam_id};

    const auto& obs_it = map.getKeyframeObs().find(tcid);
    if (obs_it == map.getKeyframeObs().end()) {
      continue;
    }
    for (const auto& lm_id : obs_it->second) unique_lms.insert(lm_id);
  }

  size_t num_unique_lms = unique_lms.size();
  float total_observed_lms_score = num_unique_lms / (num_unique_lms + 10.0f);

  return total_observed_lms_score;
}

float MapDatabase::computeTimeScore(FrameId kf_id, FrameId t_min, float t_range) {
  float t_norm = (t_range > 0.0) ? static_cast<float>((kf_id - t_min) / t_range) : 0.5f;
  float time_score = std::abs(2.0f * t_norm - 1.0f);

  return time_score;
}

float MapDatabase::computeKeyframeScore(const KeyframeScore& kf_score) {
  float score = 0.0f;

  score += 0.4f * kf_score.node_score.graph_score;
  score += 0.2f * kf_score.node_score.loop_score;
  score += 0.1f * kf_score.time_score;
  score += 0.1f * kf_score.img_bbox_coverage;
  score += 0.2f * kf_score.total_observed_lms_score;

  return score;
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

}  // namespace basalt
