#include <basalt/vi_estimator/loop_closing.h>
#include "basalt/optical_flow/optical_flow.h"
#include "basalt/utils/common_types.h"
#include "basalt/utils/eigen_utils.hpp"
#include "basalt/utils/keypoints.h"
#include "basalt/utils/lc_time_utils.h"
#include "basalt/vi_estimator/landmark_database.h"
#include "basalt/vi_estimator/map_interface.h"

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <Eigen/Core>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <sophus/se3.hpp>
#include <vector>

namespace basalt {

LoopClosing::LoopClosing(const VioConfig& config, const Calibration<double>& calib) {
  this->config = config;
  this->calib = calib;
  hash_bow_database.reset(new HashBow<256>(config.loop_closing_bow_num_bits));
}

void LoopClosing::initialize() {
  auto proc_func = [&]() {
    LoopClosingInput::Ptr loop_closing_input;
    int iters_left_to_close_loop = config.loop_closing_frequency;

    bool notify_lc_finished = false;
    std::ofstream dump_loop_detection_result_file("loop_detection_times.csv", std::ios::out);
    // Header
    lc_time_stats.dumpHeader(dump_loop_detection_result_file);
    lc_time_stats.resetStats();

    while (true) {
      if (lc_time_stats.current_kf_ts != -1) {
        dump_loop_detection_result_file << lc_time_stats;
        lc_time_stats.resetStats();
      }

      if (notify_lc_finished && sync_lc_finished != nullptr) {
        std::lock_guard<std::mutex> lk(sync_lc_finished->m);
        sync_lc_finished->ready = true;
        sync_lc_finished->cvar.notify_one();
        notify_lc_finished = false;
      }

      in_optical_flow_queue.pop(loop_closing_input);
      if (loop_closing_input == nullptr) {
        if (out_lc_vis_queue) out_lc_vis_queue->push(nullptr);
        break;
      }

      lc_time_stats.addTime(LCTimeStage::Start, true);
      lc_time_stats.current_kf_ts = loop_closing_input->t_ns;

      updateHashBowDatabase(loop_closing_input);
      lc_time_stats.addTime(LCTimeStage::HashBowIndex, true);

      if (config.loop_closing_timestamps.size() > 0) {
        if (config.loop_closing_timestamps.front() < loop_closing_input->t_ns) {
          close_loop = true;
          config.loop_closing_timestamps.erase(config.loop_closing_timestamps.begin());
        }
      }

      if (!close_loop && !config.always_detect_loop) {
        notify_lc_finished = true;
        continue;
      }

      TimeCamId best_candidate_tcid;
      Sophus::SE3f best_corrected_pose;
      std::vector<FrameId> best_island;
      std::unordered_map<LandmarkId, LandmarkId> lm_fusions;
      std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Vec2>> curr_lc_obs;
      bool success = runLoopClosure(loop_closing_input, best_candidate_tcid, best_corrected_pose, best_island,
                                    lm_fusions, curr_lc_obs);

      if (!success) {
        lc_time_stats.loop_closed = false;
        notify_lc_finished = true;
        loop_closing_visualization_data->loop_closure_found = false;
        if (out_lc_vis_queue) {
          out_lc_vis_queue->push(loop_closing_visualization_data);
        }
        continue;
      }

      lc_time_stats.loop_closed = true;

      auto map_msg = std::make_shared<ReadMapReqMsg>();
      map_msg->frame_id = loop_closing_input->t_ns;
      out_map_req_queue->push(map_msg);
      in_map_res_queue.pop(map_response);

      // get the distance between best_corrected_pose and the current pose
      Sophus::SE3f current_pose = map_response->keyframe_poses->at(loop_closing_input->t_ns).cast<float>();
      float drift_reduced = (current_pose.translation() - best_corrected_pose.translation()).norm();
      if (drift_reduced < config.loop_closing_min_drift_reduction) {
        notify_lc_finished = true;
        loop_closing_visualization_data->loop_closure_found = false;
        if (out_lc_vis_queue) {
          out_lc_vis_queue->push(loop_closing_visualization_data);
        }
        continue;
      }

      if (out_lc_vis_queue) {
        loop_closing_visualization_data->keyframe_pose =
            map_response->keyframe_poses->at(loop_closing_input->t_ns).cast<float>();
        for (const auto& kf_id : best_island) {
          loop_closing_visualization_data->candidate_pose[kf_id] =
              map_response->keyframe_poses->at(kf_id).cast<float>();
        }
      }

      if (config.close_loops) {
        success = closeLoop(loop_closing_input->t_ns, best_island, best_corrected_pose, lm_fusions, curr_lc_obs);
        lc_time_stats.addTime(LCTimeStage::LoopClosure, true);

        if (out_lc_vis_queue) {
          loop_closing_visualization_data->loop_closure_found = success;
          out_lc_vis_queue->push(loop_closing_visualization_data);
        }

        if (success) {
          close_loop = false;
        }
      } else {
        if (out_lc_vis_queue) {
          loop_closing_visualization_data->loop_closure_found = success;
          out_lc_vis_queue->push(loop_closing_visualization_data);
        }

        notify_lc_finished = true;
      }

      if (!success) notify_lc_finished = true;
    }
  };

  std::ofstream dump_loop_detection_result_file("loop_detection_results.csv", std::ios::out);
  if (dump_loop_detection_result_file.is_open()) {
    dump_loop_detection_result_file
        << "curr_kf_id,candidate_kf_id,num_matches,num_inliers,curr_tx,curr_ty,curr_tz,curr_qx,"
           "curr_qy,curr_qz,curr_qw,cand_tx,cand_ty,cand_tz,cand_qx,cand_qy,"
           "cand_qz,cand_qw,corr_tx,corr_ty,corr_tz,corr_qx,corr_qy,corr_qz,"
           "corr_qw"
        << std::endl;
    dump_loop_detection_result_file.close();
  } else {
    std::cerr << "Failed to create loop_detection_results.csv for writing." << std::endl;
  }

  processing_thread.reset(new std::thread(proc_func));
}

bool LoopClosing::closeLoop(const FrameId curr_kf_id, const std::vector<FrameId>& best_island,
                            const Sophus::SE3f& best_corrected_pose,
                            const std::unordered_map<LandmarkId, LandmarkId>& lm_fusions,
                            const std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Vec2>>& curr_lc_obs) {
  MapOfPoses map_of_poses;
  VectorOfConstraints constraints;
  buildPoseGraph(TimeCamId{curr_kf_id, 0}, best_island, best_corrected_pose, map_of_poses, constraints);

  ceres::Problem problem;
  buildOptimizationProblem(constraints, &map_of_poses, &problem);

  bool success = solveOptimizationProblem(&problem);

  if (!success) {
    return false;
  }

  restorePosesFromCeres(map_of_poses);

  auto map_update_msg = std::make_shared<WriteMapUpdateMsg>();
  map_update_msg->keyframe_poses = map_response->keyframe_poses;
  map_update_msg->candidate_kf_id = best_island[0];
  map_update_msg->curr_kf_id = curr_kf_id;
  map_update_msg->lm_fusions = lm_fusions;
  map_update_msg->curr_lc_obs = curr_lc_obs;
  if (out_map_update_queue) out_map_update_queue->push(map_update_msg);

  return true;
}

bool LoopClosing::runLoopClosure(const LoopClosingInput::Ptr& loop_closing_input, TimeCamId& best_candidate_tcid,
                                 Sophus::SE3f& best_corrected_pose, std::vector<FrameId>& best_island,
                                 std::unordered_map<LandmarkId, LandmarkId>& lm_fusions,
                                 std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Vec2>>& curr_lc_obs) {
  FrameId curr_kf_id = loop_closing_input->t_ns;

  if (out_lc_vis_queue) {
    loop_closing_visualization_data = std::make_shared<LoopClosingVisualizationData>();
    loop_closing_visualization_data->t_ns = curr_kf_id;
  }

  std::vector<Eigen::aligned_unordered_map<KeypointId, Keypoint>> curr_kf_kpts(
      calib.intrinsics.size());  // cambiar el tipo de esto. Keypoint no hace falta
  std::vector<std::bitset<256>> descriptors;

  if (config.loop_closing_use_all_recent_keypoints) {
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      for (const auto& kpt : loop_closing_input->keypoints[i]) {
        curr_kf_kpts[i][kpt.first] = kpt.second;
        descriptors.emplace_back(kpt_descriptors[TimeCamId{curr_kf_id, i}][kpt.first]);
      }
    }
  } else {
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      /*
      TimeCamId tcid = TimeCamId{curr_kf_id, i};
      auto it = map->getKeyframeObs().find(tcid);
      if (it == map->getKeyframeObs().end()) {
        continue;
      }
      std::set<LandmarkId> lm_ids = it->second;
      for (const auto& lm_id : lm_ids) {
        auto lm = map->getLandmark(lm_id);
        auto kpt_it = lm.obs.find(tcid);
        if (kpt_it != lm.obs.end()) {
          curr_kf_kpts[i][lm_id] = Eigen::Translation2f(kpt_it->second.cast<float>());
        }
      }
      */
      for (const auto& kpt : loop_closing_input->landmarks[i]) {
        curr_kf_kpts[i][kpt.first] = kpt.second;
        descriptors.emplace_back(kpt_descriptors[TimeCamId{curr_kf_id, i}][kpt.first]);
      }
    }
  }

  if (curr_kf_kpts.empty()) return false;  // chequear que esto esté bien

  /*
    if (config.loop_closing_query_with_all_cameras) {
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        for (const auto& kpt : curr_kf_kpts[i]) {
          descriptors.emplace_back(kpt_descriptors[TimeCamId{curr_kf_id, i}][kpt.first]);
        }
      }
    } else {
      descriptors.reserve(curr_kf_kpts[0].size());
      for (const auto& kpt : curr_kf_kpts[0]) {
        descriptors.emplace_back(kpt_descriptors[TimeCamId{curr_kf_id, 0}][kpt.first]);
      }
    }
  */

  HashBowVector bow_vector;
  std::vector<FeatureHash> hashes;
  // TODO: maybe i can reuse the bow that was previously computed and indexed
  hash_bow_database->compute_bow(descriptors, hashes, bow_vector);

  std::vector<FrameId> candidate_kfs;
  std::vector<double> candidate_scores;
  query_hashbow_database(curr_kf_id, bow_vector, candidate_kfs, candidate_scores);
  lc_time_stats.addTime(LCTimeStage::HashBowSearch, true);

  if (candidate_kfs.empty()) return false;

  if (out_lc_vis_queue) {
    loop_closing_visualization_data->hashbow_results = candidate_kfs;
    loop_closing_visualization_data->hashbow_scores = candidate_scores;
  }

  bool loop_found = false;
  for (const auto& candidate_kf : candidate_kfs) {
    std::unordered_map<FrameId, std::vector<KeyframesMatch>> matches;
    std::unordered_map<FrameId, std::vector<KeyframesMatch>> inlier_matches;
    size_t initial_matches, initial_island_matches, initial_inliers, reprojection_inliers;

    if (config.loop_closing_skip_covisible && are_covisible(curr_kf_id, candidate_kf)) continue;

    std::vector<std::vector<KeypointId>> candidate_kf_kpts(calib.intrinsics.size());
    if (!config.loop_closing_skip_initial_matching) {
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        TimeCamId candidate_tcid{candidate_kf, i};
        for (const auto& kpt : kpt_descriptors[candidate_tcid]) {
          candidate_kf_kpts[i].emplace_back(kpt.first);
        }
      }
      match_keyframe(curr_kf_id, curr_kf_kpts, candidate_kf_kpts, candidate_kf, matches[candidate_kf]);

      if (matches[candidate_kf].size() < config.loop_closing_min_initial_matches) {
        candidate_kf_kpts.clear();
        matches[candidate_kf].clear();

        lc_time_stats.addTime(LCTimeStage::InitialMatch, false);
        continue;
      }

      candidate_kf_kpts.clear();
      matches[candidate_kf].clear();
    }
    lc_time_stats.addTime(LCTimeStage::InitialMatch, true);

    // get the 3d points from the map database
    auto msg = std::make_shared<Read3dPointsReqMsg>();
    msg->keyframe = candidate_kf;
    msg->neighbors_num = config.loop_closing_num_neighbors;
    out_map_req_queue->push(msg);

    MapIslandResponse::Ptr map_island = std::make_shared<MapIslandResponse>();
    in_map_3d_points_queue.pop(map_island);
    std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>> points_3d = map_island->landmarks_3d_map;

    std::vector<FrameId> kfs_island = {candidate_kf};
    std::vector<FrameId> neighboring_kfs = map_island->island_keyframes;
    kfs_island.insert(kfs_island.end(), neighboring_kfs.begin(), neighboring_kfs.end());

    lc_time_stats.addTime(LCTimeStage::LandmarksRequest, true);

    candidate_kf_kpts.resize(calib.intrinsics.size());
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      TimeCamId candidate_tcid{candidate_kf, i};
      if (points_3d.find(candidate_tcid) == points_3d.end()) {
        continue;
      }
      for (const auto& point_3d : points_3d.at(candidate_tcid)) {
        candidate_kf_kpts[i].emplace_back(point_3d.first);
      }
    }
    match_keyframe(curr_kf_id, curr_kf_kpts, candidate_kf_kpts, candidate_kf, matches[candidate_kf]);

    if (matches[candidate_kf].size() < static_cast<size_t>(config.loop_closing_min_matches)) {
      lc_time_stats.addTime(LCTimeStage::IslandMatch, false);
      continue;
    }

    initial_matches = matches[candidate_kf].size();

    for (const auto& neighbor_kf : neighboring_kfs) {
      std::vector<std::vector<KeypointId>> candidate_kf_kpts(calib.intrinsics.size());
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        TimeCamId candidate_tcid{neighbor_kf, i};
        if (points_3d.find(candidate_tcid) == points_3d.end()) {
          continue;
        }
        for (const auto& point_3d : points_3d.at(candidate_tcid)) {
          candidate_kf_kpts[i].emplace_back(point_3d.first);
        }
      }
      // TODO: here, skip the source kpts that were already matched previously
      match_keyframe(curr_kf_id, curr_kf_kpts, candidate_kf_kpts, neighbor_kf, matches[neighbor_kf]);
    }

    initial_island_matches = 0;
    for (const auto& kf_id : neighboring_kfs) {
      initial_island_matches += matches[kf_id].size();
    }

    lc_time_stats.addTime(LCTimeStage::IslandMatch, true);

    Sophus::SE3d absolute_pose;
    size_t num_inliers = computeAbsolutePoseMultiCam(curr_kf_id, kfs_island, curr_kf_kpts, points_3d, matches,
                                                     inlier_matches, absolute_pose);

    if (num_inliers < config.loop_closing_pnp_min_inliers) {
      lc_time_stats.addTime(LCTimeStage::GeometricVerification, false);

      continue;
    }

    initial_inliers = 0;
    for (const auto& kf_id : kfs_island) {
      initial_inliers += inlier_matches[kf_id].size();
    }
    lc_time_stats.addTime(LCTimeStage::GeometricVerification, true);

    // reproject the 3d points from the kfs_island[0] to the current keyframe's left cam
    std::vector<Eigen::aligned_unordered_map<KeypointId, Vec2>> reprojected_keypoints(
        config.loop_closing_cameras_to_reproject.size());
    std::vector<std::unordered_map<KeypointId, Eigen::aligned_vector<Vec2>>> redetected_keypoints_map(
        config.loop_closing_cameras_to_reproject.size());
    std::vector<Eigen::aligned_vector<Vec2>> rematched_keypoints(config.loop_closing_cameras_to_reproject.size());
    for (size_t i = 0; i < config.loop_closing_cameras_to_reproject.size(); i++) {
      size_t cam_id = config.loop_closing_cameras_to_reproject[i];

      if (points_3d.find(TimeCamId{candidate_kf, cam_id}) == points_3d.end()) {
        continue;
      }
      reproject_landmarks(absolute_pose, kfs_island[0], reprojected_keypoints[i], cam_id,
                          points_3d.at(TimeCamId{candidate_kf, cam_id}));
      std::vector<KeyframesMatch> new_matched_keypoints;
      for (const auto& kp : reprojected_keypoints[i]) {
        Eigen::aligned_vector<Vec2>& redetected_kpts = redetected_keypoints_map[i][kp.first];

        Vec2 best_keypoint_pos;

        std::bitset<256> center_kpt_descriptor = kpt_descriptors[TimeCamId{candidate_kf, cam_id}][kp.first];
        bool match_found = redetect_kpts(kp.second, center_kpt_descriptor, redetected_kpts,
                                         *loop_closing_input->input_images->img_data[cam_id].img, best_keypoint_pos);

        if (match_found) {
          rematched_keypoints[i].emplace_back(best_keypoint_pos);
          if (config.loop_closing_use_rematches) {
            new_matched_keypoints.emplace_back(KeyframesMatch{
                cam_id, cam_id, kp.first, kp.first});  // source_cam, source_kpt_id, target_cam, target_kpt_id
            curr_kf_kpts[cam_id][kp.first] = Eigen::Translation2f(best_keypoint_pos);
          }
        }
      }

      // filter the new matches, discarding those that contain an existing taget_kpt_id
      if (config.loop_closing_use_rematches) {
        std::set<KeypointId> existing_target_kpt_ids;
        for (const auto& keyframe_match : matches) {
          for (const auto& match : keyframe_match.second) {
            if (match.source_cam != cam_id) continue;  // only consider matches from the current camera
            existing_target_kpt_ids.insert(match.target_kpt_id);
          }
        }

        std::vector<KeyframesMatch> filtered_new_matched_keypoints;
        for (const auto& match : new_matched_keypoints) {
          if (existing_target_kpt_ids.find(match.target_kpt_id) == existing_target_kpt_ids.end()) {
            filtered_new_matched_keypoints.emplace_back(match);
          }
        }

        // add the new matches to the matches
        matches[candidate_kf].insert(matches[candidate_kf].end(), filtered_new_matched_keypoints.begin(),
                                     filtered_new_matched_keypoints.end());
      }
    }
    lc_time_stats.addTime(LCTimeStage::Reprojection, true);

    if (config.loop_closing_use_rematches) {
      // recompute the absolute pose with the new matches
      inlier_matches.clear();
      size_t new_num_inliers = computeAbsolutePoseMultiCam(curr_kf_id, kfs_island, curr_kf_kpts, points_3d, matches,
                                                           inlier_matches, absolute_pose);

      lc_time_stats.addTime(LCTimeStage::ReprojectedGeometricVerification, true);

      reprojection_inliers = new_num_inliers;

      std::cout << curr_kf_id << " -> " << candidate_kf << ": initial matches = " << initial_matches
                << ", island matches = " << initial_island_matches << ", initial inliers = " << initial_inliers
                << ", reprojection inliers = " << reprojection_inliers << std::endl;
    } else {
      reprojection_inliers = num_inliers;

      std::cout << curr_kf_id << " -> " << candidate_kf << ": initial matches = " << initial_matches
                << ", island matches = " << initial_island_matches << ", initial inliers = " << initial_inliers
                << std::endl;
    }

    // Fuse the landmarks
    for (const auto& [kf_id, inlier_matches_vec] : inlier_matches) {
      for (const auto& match : inlier_matches_vec) {
        LandmarkId candidate_lm_id = match.target_kpt_id;
        LandmarkId curr_lm_id = match.source_kpt_id;

        if (lm_fusions.find(candidate_lm_id) == lm_fusions.end()) {
          lm_fusions[curr_lm_id] = candidate_lm_id;
        }

        TimeCamId curr_tcid{curr_kf_id, match.source_cam};
        curr_lc_obs[curr_tcid][curr_lm_id] = curr_kf_kpts[match.source_cam][curr_lm_id].translation().cast<float>();
      }
    }

    if (out_lc_vis_queue) {
      loop_closing_visualization_data->islands.emplace_back(kfs_island);
      loop_closing_visualization_data->corrected_pose[candidate_kf] = absolute_pose.cast<float>();
      loop_closing_visualization_data->candidate_kfs.emplace_back(candidate_kf);
      loop_closing_visualization_data->reprojected_keypoints.emplace_back(reprojected_keypoints);
      loop_closing_visualization_data->redetected_keypoints.emplace_back(redetected_keypoints_map);
      loop_closing_visualization_data->rematched_keypoints.emplace_back(rematched_keypoints);

      // set the current keypoints
      loop_closing_visualization_data->current_keypoints.resize(calib.intrinsics.size());
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        for (const auto& kpt : curr_kf_kpts[i]) {
          loop_closing_visualization_data->current_keypoints[i].emplace_back(kpt.second.translation().cast<float>());
        }
      }

      std::vector<Eigen::aligned_unordered_map<TimeCamId, Eigen::aligned_vector<std::pair<Vec2, Vec2>>>>
          curr_island_matches(calib.intrinsics.size());

      std::vector<Eigen::aligned_unordered_map<TimeCamId, Eigen::aligned_vector<std::pair<Vec2, Vec2>>>>
          curr_island_inliers(calib.intrinsics.size());
      Eigen::aligned_vector<Eigen::Vector3d> curr_island_landmarks;
      std::vector<LandmarkId> curr_island_landmark_ids;

      for (const auto& kf_id : kfs_island) {
        // first, set the keypoints for each camera
        for (size_t i = 0; i < calib.intrinsics.size(); i++) {
          TimeCamId kf_tcid = TimeCamId{kf_id, i};

          loop_closing_visualization_data->candidate_keypoints[kf_tcid] = Eigen::aligned_vector<Vec2>();

          if (points_3d.find(kf_tcid) == points_3d.end()) {
            continue;
          }
          for (const auto& lms : points_3d.at(kf_tcid)) {
            auto& positions = kpts_positions[kf_tcid];
            if (positions.find(lms.first) == positions.end()) {
              std::cout << "1. Landmark " << lms.first << " does not have a keypoint position in the map." << std::endl;
              continue;
            }
            loop_closing_visualization_data->candidate_keypoints[kf_tcid].emplace_back(
                positions[lms.first].cast<float>());
          }
        }

        // set the matches of the kf_id
        // loop_closing_visualization_data->matches.resize(calib.intrinsics.size());
        for (const auto& match : matches[kf_id]) {
          Vec2 p1 = curr_kf_kpts[match.source_cam][match.source_kpt_id].translation().cast<float>();
          //          Vec2 p2 = map->getLandmark(match.target_kpt_id).obs.at(TimeCamId{kf_id,
          //          match.target_cam}).cast<float>();
          Vec2 p2 = kpts_positions[TimeCamId{kf_id, match.target_cam}][match.target_kpt_id].cast<float>();
          curr_island_matches[match.source_cam][TimeCamId{kf_id, match.target_cam}].emplace_back(p1, p2);
        }

        // set the inlier matches of the kf_id
        for (const auto& match : inlier_matches[kf_id]) {
          Vec2 p1 = curr_kf_kpts[match.source_cam][match.source_kpt_id].translation().cast<float>();
          //          if (!map->landmarkExists(match.target_kpt_id)) {
          //            std::cout << "3. Landmark " << match.target_kpt_id << " does not exist in the map." <<
          //            std::endl; continue;
          //          }
          //
          //          Vec2 p2 = map->getLandmark(match.target_kpt_id).obs.at(TimeCamId{kf_id,
          //          match.target_cam}).cast<float>();
          Vec2 p2 = kpts_positions[TimeCamId{kf_id, match.target_cam}][match.target_kpt_id].cast<float>();
          curr_island_inliers[match.source_cam][TimeCamId{kf_id, match.target_cam}].emplace_back(p1, p2);
        }

        for (const auto& match : inlier_matches[kf_id]) {
          /*
                    if (!map->landmarkExists(match.target_kpt_id)) {
                      std::cout << "4. Landmark " << match.target_kpt_id << " does not exist in the map." << std::endl;
                      continue;
                    }

                    const auto& lm = map->getLandmark(match.target_kpt_id);

                    TimeCamId host_kf_id = lm.host_kf_id;
                    Sophus::SE3f T_w_i = map->getKeyframePose(host_kf_id.frame_id);  // keyframe pose
                    Sophus::SE3f T_i_c = calib.T_i_c[host_kf_id.cam_id].cast<float>();
                    Sophus::SE3f T_w_camera = T_w_i * T_i_c;

                    Vec4f pt_c = StereographicParam<Scalar>::unproject(lm.direction);
                    pt_c *= 1 / lm.inv_dist;
                    pt_c[3] = 1;
                    Vec4f pt_w = T_w_camera * pt_c;  // homogeneous world point
                    Eigen::Vector3d world_pt = pt_w.head<3>().template cast<double>();
          */

          const auto& target_tcid = TimeCamId{kf_id, match.target_cam};
          const auto& world_pt = points_3d.at(target_tcid).at(match.target_kpt_id);

          curr_island_landmarks.emplace_back(world_pt);
          curr_island_landmark_ids.emplace_back(match.target_kpt_id);
        }
      }
      loop_closing_visualization_data->landmarks.emplace_back(curr_island_landmarks);
      loop_closing_visualization_data->landmark_ids.emplace_back(curr_island_landmark_ids);
      loop_closing_visualization_data->matches.emplace_back(curr_island_matches);
      loop_closing_visualization_data->inlier_matches.emplace_back(curr_island_inliers);
    }

    if (config.dump_loop_detection_result) {
      Sophus::SE3d corrected_pose = absolute_pose;
      Sophus::SE3f curr_pose = map->getKeyframePose(curr_kf_id);
      Sophus::SE3f candidate_pose = map->getKeyframePose(candidate_kf);
      Eigen::Quaternionf curr_quat(curr_pose.unit_quaternion());
      Eigen::Quaternionf candidate_quat(candidate_pose.unit_quaternion());
      Eigen::Quaternionf corrected_quat(corrected_pose.unit_quaternion());

      size_t num_matches = 0;
      size_t num_inliers = 0;
      for (const auto& kf_id : kfs_island) {
        num_matches += matches[kf_id].size();
        num_inliers += inlier_matches[kf_id].size();
      }

      std::ofstream dump_loop_detection_result_file("loop_detection_results.csv", std::ios::app);
      if (dump_loop_detection_result_file.is_open()) {
        dump_loop_detection_result_file << curr_kf_id << "," << candidate_kf << "," << num_matches << "," << num_inliers
                                        << "," << curr_pose.translation().x() << "," << curr_pose.translation().y()
                                        << "," << curr_pose.translation().z() << "," << curr_quat.x() << ","
                                        << curr_quat.y() << "," << curr_quat.z() << "," << curr_quat.w() << ","
                                        << candidate_pose.translation().x() << "," << candidate_pose.translation().y()
                                        << "," << candidate_pose.translation().z() << "," << candidate_quat.x() << ","
                                        << candidate_quat.y() << "," << candidate_quat.z() << "," << candidate_quat.w()
                                        << "," << corrected_pose.translation().x() << ","
                                        << corrected_pose.translation().y() << "," << corrected_pose.translation().z()
                                        << "," << corrected_quat.x() << "," << corrected_quat.y() << ","
                                        << corrected_quat.z() << "," << corrected_quat.w() << std::endl;

        dump_loop_detection_result_file.close();
      } else {
        std::cerr << "Failed to open loop_detection_results.csv for writing." << std::endl;
      }
    }
    best_candidate_tcid = TimeCamId{candidate_kf, 0};
    best_corrected_pose = absolute_pose.cast<float>();
    best_island = kfs_island;
    loop_found = true;
    break;
  }

  return loop_found;
}

void LoopClosing::filter_matches(std::vector<FrameId>& kfs_island,
                                 std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches) {
  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    std::set<KeypointId> matched_kpts;

    for (const auto& kf_id : kfs_island) {
      std::vector<KeyframesMatch>& kf_matches = matches[kf_id];
      std::vector<KeyframesMatch> filtered_matches;

      for (const auto& match : kf_matches) {
        if (matched_kpts.find(match.source_kpt_id) == matched_kpts.end() && match.source_cam == i) {
          filtered_matches.emplace_back(match);
          matched_kpts.emplace(match.source_kpt_id);
        }
      }

      kf_matches = filtered_matches;
    }
  }
}

void LoopClosing::query_hashbow_database(FrameId curr_kf_id, HashBowVector& bow_vector, std::vector<FrameId>& results,
                                         std::vector<double>& scores) {
  results.clear();

  std::vector<std::pair<TimeCamId, double>> query_results;
  // max_t_ns should be the current keyframe minus some margin to avoid matching with future frames
  int64_t max_t_ns = static_cast<int64_t>(curr_kf_id) - config.loop_closing_frame_time_margin_s * 1e9;
  hash_bow_database->querry_database(bow_vector, config.loop_closing_num_frames_to_match, query_results, &max_t_ns);

  for (const auto& [candidate_kf_tcid, score] : query_results) {
    if (candidate_kf_tcid.frame_id == curr_kf_id) continue;
    if (score < config.loop_closing_frames_to_match_threshold) continue;

    if (std::find(results.begin(), results.end(), candidate_kf_tcid.frame_id) != results.end()) continue;
    results.emplace_back(candidate_kf_tcid.frame_id);
    scores.emplace_back(score);
  }
}

void LoopClosing::discard_covisible_keyframes(const FrameId& curr_kf_id, std::vector<FrameId>& candidate_kfs) {
  std::set<LandmarkId> landmarks;
  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    TimeCamId tcid = TimeCamId{curr_kf_id, i};
    auto it = map->getKeyframeObs().find(tcid);
    if (it != map->getKeyframeObs().end()) {
      for (const auto& lm_id : it->second) {
        landmarks.emplace(lm_id);
      }
    }
  }

  std::vector<FrameId> filtered_similar_kfs;
  for (const auto& candidate_kf_id : candidate_kfs) {
    std::set<LandmarkId> candidate_kf_landmarks;
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      TimeCamId tcid = TimeCamId{candidate_kf_id, i};
      auto it = map->getKeyframeObs().find(tcid);
      if (it != map->getKeyframeObs().end()) {
        for (const auto& lm_id : it->second) {
          candidate_kf_landmarks.emplace(lm_id);
        }
      }
    }

    std::vector<LandmarkId> common;
    std::set_intersection(candidate_kf_landmarks.begin(), candidate_kf_landmarks.end(), landmarks.begin(),
                          landmarks.end(), std::back_inserter(common));
    if (common.empty()) filtered_similar_kfs.emplace_back(candidate_kf_id);
  }

  candidate_kfs = filtered_similar_kfs;
}

bool LoopClosing::redetect_kpts(const Vec2& center_kpt, std::bitset<256>& center_kpt_descriptor,
                                Eigen::aligned_vector<Vec2>& redetected_kpts, const basalt::ManagedImage<uint16_t>& img,
                                Vec2& best_keypoint_pos) {
  if (!img.InBounds(center_kpt.x(), center_kpt.y(), config.loop_closing_fast_grid_size / 2.0)) return false;

  const basalt::Image<const uint16_t>& sub_img_raw =
      img.SubImage(center_kpt.x() - (config.loop_closing_fast_grid_size / 2.0),
                   center_kpt.y() - (config.loop_closing_fast_grid_size / 2.0), config.loop_closing_fast_grid_size,
                   config.loop_closing_fast_grid_size);

  const basalt::Image<const uint16_t>& img_raw = img.SubImage(0, 0, img.w, img.h);

  KeypointsData kd;
  detectKeypointsFAST(sub_img_raw, kd, config.loop_closing_fast_threshold, config.loop_closing_fast_nonmax_suppression);

  for (size_t i = 0; i < kd.corners.size(); i++) {
    kd.corners[i] += Eigen::Vector2d(center_kpt.x() - (config.loop_closing_fast_grid_size / 2.0),
                                     center_kpt.y() - (config.loop_closing_fast_grid_size / 2.0));
  }

  for (const auto& kpt : kd.corners) {
    redetected_kpts.emplace_back(kpt.cast<float>());
  }

  computeAngles(img_raw, kd, true);
  computeDescriptors(img_raw, kd);

  std::vector<std::pair<int, int>> matches;
  std::vector<std::bitset<256>> target_descriptor_vec(1);
  target_descriptor_vec[0] = center_kpt_descriptor;
  matchDescriptors(target_descriptor_vec, kd.corner_descriptors, matches,
                   config.loop_closing_redetect_max_hamming_distance,
                   config.loop_closing_redetect_second_best_test_ratio);

  if (matches.size() > 0) {
    best_keypoint_pos = kd.corners[matches[0].second].cast<float>();
    return true;
  }

  return false;
}

void LoopClosing::reproject_landmarks(const Sophus::SE3d& absolute_pose, FrameId candidate_kf,
                                      Eigen::aligned_unordered_map<KeypointId, Vec2>& reprojected_keypoints,
                                      size_t cam_id, const Eigen::aligned_map<LandmarkId, Vec3d>& points_3d) {
  // get the landmarks observed in candidate_kf's left cam
  std::set<LandmarkId> candidate_kf_landmarks;
  TimeCamId candidate_kf_tcid = TimeCamId{candidate_kf, cam_id};

  for (const auto& point : points_3d) {
    candidate_kf_landmarks.emplace(point.first);
  }

  Sophus::SE3d T_i_cl = calib.T_i_c[cam_id];
  Sophus::SE3d T_w_cl = absolute_pose * T_i_cl;  // world → left cam
  Sophus::SE3d T_cl_w = T_w_cl.inverse();        // left cam ← world

  for (const auto& lm_id : candidate_kf_landmarks) {
    if (points_3d.find(lm_id) == points_3d.end()) continue;
    Eigen::Vector3d world_pt = points_3d.at(lm_id);

    Eigen::Vector3d pt_cl = T_cl_w * world_pt;

    Eigen::Vector2d uv;
    bool valid = calib.intrinsics[cam_id].project(pt_cl, uv);

    if (pt_cl.z() <= 0) {
      // The point is behind the camera — invalid projection
      continue;
    }

    if (!valid) continue;

    // check if the reprojected point is within the image bounds
    const size_t w = calib.resolution[cam_id].x();
    const size_t h = calib.resolution[cam_id].y();
    if (uv.x() < 0 || uv.x() >= w || uv.y() < 0 || uv.y() >= h) {
      continue;
    }

    reprojected_keypoints[lm_id] = uv.cast<float>();
  }
}

bool LoopClosing::are_covisible(const FrameId& kf1_id, const FrameId& kf2_id) {
  std::unordered_set<KeypointId> kf1_landmarks;  // Use unordered_set for faster lookups
  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    TimeCamId tcid = TimeCamId{kf1_id, i};
    const auto& kf1_kpt_descriptors = kpt_descriptors[tcid];
    for (const auto& kpt_desc : kf1_kpt_descriptors) {
      kf1_landmarks.insert(kpt_desc.first);
    }
  }

  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    TimeCamId tcid = TimeCamId{kf2_id, i};
    const auto& kf2_kpt_descriptors = kpt_descriptors[tcid];
    for (const auto& kpt_desc : kf2_kpt_descriptors) {
      if (kf1_landmarks.count(kpt_desc.first)) {  // Faster lookup with unordered_set
        return true;
      }
    }
  }

  return false;
}

void LoopClosing::match_keyframe(const FrameId& curr_kf_id,
                                 const std::vector<Eigen::aligned_unordered_map<KeypointId, Keypoint>>& curr_kf_kpts,
                                 const std::vector<std::vector<KeypointId>>& candidate_kf_kpts,
                                 const FrameId& candidate_kf_id, std::vector<KeyframesMatch>& matches) {
  matches.clear();
  std::vector<std::vector<std::bitset<256>>> curr_kf_descriptors(calib.intrinsics.size());
  std::vector<std::vector<KeypointId>> keypoints_vectors(calib.intrinsics.size());
  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    curr_kf_descriptors[i].reserve(curr_kf_kpts[i].size());
    for (const auto& kpt : curr_kf_kpts[i]) {
      curr_kf_descriptors[i].emplace_back(kpt_descriptors[TimeCamId{curr_kf_id, i}][kpt.first]);
      keypoints_vectors[i].emplace_back(kpt.first);
    }
  }

  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    for (size_t j = 0; j < calib.intrinsics.size(); j++) {
      TimeCamId candidate_kf_tcid = TimeCamId{candidate_kf_id, j};

      std::vector<std::bitset<256>> candidate_kf_descriptors;
      candidate_kf_descriptors.reserve(candidate_kf_kpts[j].size());
      for (const auto& lm_id : candidate_kf_kpts[j]) {
        const auto& descriptor = kpt_descriptors[candidate_kf_tcid][lm_id];
        candidate_kf_descriptors.emplace_back(descriptor);
      }

      std::vector<std::pair<int, int>> curr_matches;
      matchDescriptors(curr_kf_descriptors[i], candidate_kf_descriptors, curr_matches,
                       config.loop_closing_max_hamming_distance, config.loop_closing_second_best_test_ratio);

      for (const auto& match : curr_matches) {
        matches.emplace_back(
            KeyframesMatch{i, j, keypoints_vectors[i][match.first], candidate_kf_kpts[j][match.second]});
      }
    }
  }
}

void LoopClosing::get_neighboring_keyframes(const FrameId& kf_id, int neighbors_num,
                                            std::vector<FrameId>& neighboring_kfs) {
  // have to check this!
  neighboring_kfs.clear();

  std::map<TimeCamId, std::unordered_map<KeypointId, std::bitset<256>>>& keyframes = kpt_descriptors;

  // get neighbors_num/2 keyframes before and after kf_id
  auto it = keyframes.find(TimeCamId{kf_id, 0});
  if (it == keyframes.end()) return;
  auto it_before = it;
  auto it_after = it;
  for (int i = 0; i < neighbors_num / 2; i++) {
    if (it_before != keyframes.begin()) {
      it_before--;
      it_before--;
      neighboring_kfs.emplace_back(it_before->first.frame_id);
    }
    ++it_after;
    if (it_after != keyframes.end()) {
      ++it_after;
      neighboring_kfs.emplace_back(it_after->first.frame_id);
    }
  }
}

void LoopClosing::match_candidate_keyframes(
    const FrameId& curr_kf_id, const std::vector<Eigen::aligned_unordered_map<KeypointId, Keypoint>>& curr_kf_kpts,
    std::vector<FrameId>& candidate_kfs, std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches) {
  std::vector<std::vector<std::bitset<256>>> curr_kf_descriptors(calib.intrinsics.size());
  std::vector<std::vector<KeypointId>> keypoints_vectors(calib.intrinsics.size());
  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    curr_kf_descriptors[i].reserve(curr_kf_kpts[i].size());
    for (const auto& kpt : curr_kf_kpts[i]) {
      curr_kf_descriptors[i].emplace_back(kpt_descriptors[TimeCamId{curr_kf_id, i}][kpt.first]);
      keypoints_vectors[i].emplace_back(kpt.first);
    }
  }

  for (const auto& candidate_kf_id : candidate_kfs) {
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {    // the camera of the current kf
      for (size_t j = 0; j < calib.intrinsics.size(); j++) {  // the camera of the candidate kf
        TimeCamId candidate_kf_tcid = TimeCamId{candidate_kf_id, j};
        std::set<LandmarkId> candidate_kf_landmarks = map->getKeyframeObs().at(candidate_kf_tcid);

        std::vector<std::bitset<256>> candidate_kf_descriptors;
        std::vector<KeypointId> candidate_kf_kpts;
        candidate_kf_descriptors.reserve(candidate_kf_landmarks.size());
        for (const auto& lm_id : candidate_kf_landmarks) {
          auto lm = map->getLandmark(lm_id);
          const auto& descriptor = kpt_descriptors[candidate_kf_tcid][lm_id];
          candidate_kf_descriptors.emplace_back(descriptor);
          candidate_kf_kpts.emplace_back(lm_id);
        }

        std::vector<std::pair<int, int>> curr_matches;
        matchDescriptors(curr_kf_descriptors[i], candidate_kf_descriptors, curr_matches,
                         config.loop_closing_max_hamming_distance, config.loop_closing_second_best_test_ratio);

        for (const auto& match : curr_matches) {
          matches[candidate_kf_id].emplace_back(
              KeyframesMatch{i, j, keypoints_vectors[i][match.first], candidate_kf_kpts[match.second]});
        }
      }
    }
  }

  std::vector<FrameId> filtered_candidate_kfs;

  for (const auto& candidate_kf_id : candidate_kfs) {
    if (matches[candidate_kf_id].size() < config.loop_closing_min_matches) {
      matches.erase(candidate_kf_id);
    } else {
      filtered_candidate_kfs.emplace_back(candidate_kf_id);
    }
  }

  candidate_kfs = filtered_candidate_kfs;
}

void LoopClosing::validateLoopCandidates(
    const TimeCamId& curr_kf_tcid, const std::vector<Landmark<Scalar>>& curr_kf_landmarks,
    const std::unordered_map<TimeCamId, std::vector<Landmark<Scalar>>>& candidate_kf_landmarks_map,
    const std::vector<TimeCamId>& loop_candidates,
    const std::unordered_map<TimeCamId, std::vector<std::pair<int, int>>>& matches_map,
    std::vector<TimeCamId>& validated_candidates, std::vector<Sophus::SE3f>& corrected_poses) {
  validated_candidates.clear();
  corrected_poses.clear();

  for (const auto& candidate_kf_tcid : loop_candidates) {
    auto it = candidate_kf_landmarks_map.find(candidate_kf_tcid);
    if (it == candidate_kf_landmarks_map.end()) continue;
    const auto& candidate_kf_landmarks = it->second;

    auto matches_it = matches_map.find(candidate_kf_tcid);
    if (matches_it == matches_map.end()) continue;
    const auto& matches = matches_it->second;

    // Compute absolute pose using 2D-3D correspondences
    Sophus::SE3d absolute_pose;
    std::vector<std::pair<int, int>> inlier_matches;
    bool success = computeAbsolutePose(curr_kf_tcid, curr_kf_landmarks, candidate_kf_landmarks, matches, inlier_matches,
                                       absolute_pose);
    if (!success) continue;

    Sophus::SE3f T_i_c = calib.T_i_c[curr_kf_tcid.cam_id].cast<float>();
    if (out_lc_vis_queue) {
      Eigen::aligned_vector<std::pair<Vec2, Vec2>> inlier_matches_vec;
      for (const auto& m : inlier_matches) {
        Vec2 p1 = curr_kf_landmarks[m.first].obs.at(curr_kf_tcid);
        Vec2 p2 = candidate_kf_landmarks[m.second].obs.at(candidate_kf_tcid);
        inlier_matches_vec.emplace_back(std::make_pair(p1, p2));
      }

      Eigen::aligned_vector<std::pair<Vec2, Vec2>> matches_vec;
      for (const auto& m : matches) {
        Vec2 p1 = curr_kf_landmarks[m.first].obs.at(curr_kf_tcid);
        Vec2 p2 = candidate_kf_landmarks[m.second].obs.at(candidate_kf_tcid);
        matches_vec.emplace_back(std::make_pair(p1, p2));
      }

      // loop_closing_visualization_data->inlier_matches[0][candidate_kf_tcid] = inlier_matches_vec;
      Sophus::SE3f T_w_i = map->getKeyframePose(candidate_kf_tcid.frame_id);
      loop_closing_visualization_data->candidate_pose[candidate_kf_tcid.frame_id] = T_w_i;
      loop_closing_visualization_data->corrected_pose[candidate_kf_tcid.frame_id] =
          absolute_pose.cast<float>() * T_i_c.inverse();
    }

    validated_candidates.emplace_back(candidate_kf_tcid);
    corrected_poses.emplace_back(absolute_pose.cast<float>() * T_i_c.inverse());
  }
}

void LoopClosing::buildPoseGraph(const TimeCamId& curr_kf_tcid, const std::vector<FrameId>& best_island,
                                 const Sophus::SE3f& best_corrected_pose, MapOfPoses& map_of_poses,
                                 VectorOfConstraints& constraints) {
  map_of_poses.clear();
  constraints.clear();

  const Eigen::aligned_map<FrameId, Sophus::SE3f>& keyframe_poses = *map_response->keyframe_poses;
  const CovisibilityGraph::Ptr& covisibility_graph = map_response->covisibility_graph;

  for (const auto& [kf_id, pose] : keyframe_poses) {
    map_of_poses[kf_id] = Pose3d{pose.translation().cast<double>(), pose.unit_quaternion().cast<double>()};
  }

  // add the constraints based on the relative poses of the spanning tree
  FrameId spanning_tree_root = covisibility_graph->getRoot();
  for (const auto& [kf_id, T_w_i] : keyframe_poses) {
    if (kf_id == spanning_tree_root) continue;

    FrameId parent_kf_id = covisibility_graph->getParentNode(kf_id);
    Sophus::SE3f T_w_p = keyframe_poses.at(parent_kf_id);
    Sophus::SE3f T_p_i = T_w_p.inverse() * T_w_i;
    Constraint3d c;
    c.id_begin = parent_kf_id;
    c.id_end = kf_id;
    Pose3d relative_pose;
    relative_pose.p = T_p_i.translation().cast<double>();
    relative_pose.q = T_p_i.unit_quaternion().cast<double>();
    c.t_be = relative_pose;
    c.information = Eigen::Matrix<double, 6, 6>::Identity();
    constraints.push_back(c);
  }

  // add the past loop closure constraints
  for (const auto& [kf_id, loop_closures] : covisibility_graph->getAllLoopClosures()) {
    for (const auto& loop_kf_id : loop_closures) {
      if (kf_id < loop_kf_id) {
        Sophus::SE3f T_w_i = keyframe_poses.at(kf_id);
        Sophus::SE3f T_w_j = keyframe_poses.at(loop_kf_id);
        Sophus::SE3f T_i_j = T_w_i.inverse() * T_w_j;
        Constraint3d c;
        c.id_begin = kf_id;
        c.id_end = loop_kf_id;
        Pose3d relative_pose;
        relative_pose.p = T_i_j.translation().cast<double>();
        relative_pose.q = T_i_j.unit_quaternion().cast<double>();
        c.t_be = relative_pose;
        c.information = Eigen::Matrix<double, 6, 6>::Identity();
        constraints.push_back(c);
      }
    }
  }

  // Add constraints for the most covisible keyframes
  for (const auto& [kf_id, T_w_i] : keyframe_poses) {
    std::vector<FrameId> most_covisible_kfs =
        covisibility_graph->getAboveWeight(kf_id, config.loop_closing_pgo_min_covisibility_weight);
    for (const auto& covisible_kf_id : most_covisible_kfs) {
      if (kf_id >= covisible_kf_id) continue;

      if (kf_id != spanning_tree_root) {
        if (covisibility_graph->getParentNode(kf_id) == covisible_kf_id) {
          continue;  // already added as part of the spanning tree
        }
      }
      if (covisible_kf_id != spanning_tree_root) {
        if (covisibility_graph->getParentNode(covisible_kf_id) == kf_id) {
          continue;  // already added as part of the spanning tree
        }
      }

      Sophus::SE3f T_w_j = keyframe_poses.at(covisible_kf_id);
      Sophus::SE3f T_i_j = T_w_i.inverse() * T_w_j;
      Constraint3d c;
      c.id_begin = kf_id;
      c.id_end = covisible_kf_id;
      Pose3d relative_pose;
      relative_pose.p = T_i_j.translation().cast<double>();
      relative_pose.q = T_i_j.unit_quaternion().cast<double>();
      c.t_be = relative_pose;
      c.information = Eigen::Matrix<double, 6, 6>::Identity();
      constraints.push_back(c);
    }
  }

  // add the current loop closure constraints
  for (const auto& kf_id : best_island) {
    if (kf_id == curr_kf_tcid.frame_id) {  // shouldn't even be necessary at this point
      continue;
    }

    Sophus::SE3f T_w_i = keyframe_poses.at(kf_id);
    Sophus::SE3f T_w_j = best_corrected_pose;
    Sophus::SE3f T_i_j = T_w_i.inverse() * T_w_j;
    Constraint3d c;
    c.id_begin = kf_id;
    c.id_end = curr_kf_tcid.frame_id;
    Pose3d relative_pose;
    relative_pose.p = T_i_j.translation().cast<double>();
    relative_pose.q = T_i_j.unit_quaternion().cast<double>();
    c.t_be = relative_pose;
    c.information = Eigen::Matrix<double, 6, 6>::Identity();
    constraints.push_back(c);
  }
}

void LoopClosing::buildCeresParams(const std::vector<Sophus::SE3f>& poses,
                                   const std::vector<Sophus::SE3f>& relative_poses, MapOfPoses& map_of_poses,
                                   VectorOfConstraints& constraints) {
  map_of_poses.clear();
  constraints.clear();

  for (size_t i = 0; i < poses.size(); i++) {
    Pose3d pose;
    pose.p = poses[i].translation().cast<double>();
    pose.q = poses[i].unit_quaternion().cast<double>();
    map_of_poses[i] = pose;
  }

  for (size_t i = 0; i < relative_poses.size(); i++) {
    Constraint3d c;
    c.id_begin = i;
    c.id_end = (i + 1) % poses.size();
    Pose3d relative_pose;
    relative_pose.p = relative_poses[i].translation().cast<double>();
    relative_pose.q = relative_poses[i].unit_quaternion().cast<double>();
    c.t_be = relative_pose;
    c.information = Eigen::Matrix<double, 6, 6>::Identity();
    constraints.push_back(c);
  }
}

void LoopClosing::restorePosesFromCeres(const MapOfPoses& map_of_poses) {
  for (const auto& [id, updated_pose] : map_of_poses) {
    if (map_response->keyframe_poses->find(id) == map_response->keyframe_poses->end()) {
      std::cerr << "Error: Keyframe ID " << id << " not found in map_response->keyframe_poses." << std::endl;
      continue;
    }
    Sophus::SE3f& pose = map_response->keyframe_poses->at(id);
    pose.translation() = updated_pose.p.cast<float>();
    pose.so3() = Sophus::SO3f(updated_pose.q.cast<float>());
  }
}

void LoopClosing::updateMap(const std::vector<Sophus::SE3f>& poses, const TimeCamId& best_candidate_tcid) {
  // transform the poses before itStart to align with the optimized poses
  Sophus::SE3f T_w_correction = poses[0] * map_response->keyframe_poses->at(best_candidate_tcid.frame_id).inverse();
  auto itStart2 = map_response->keyframe_poses->lower_bound(best_candidate_tcid.frame_id);
  for (auto it = map_response->keyframe_poses->begin(); it != itStart2; it++) {
    Sophus::SE3f& old_pose = map_response->keyframe_poses->at(it->first);
    old_pose = T_w_correction * old_pose;
  }

  // update the poses of the keyframes involved in the loop closure
  auto itStart = map_response->keyframe_poses->lower_bound(best_candidate_tcid.frame_id);
  for (size_t i = 0; i < poses.size(); i++) {
    Sophus::SE3f& old_pose = map_response->keyframe_poses->at(itStart->first);
    old_pose = poses[i];
    itStart++;
  }
}

void LoopClosing::triggerLoopClosure() { close_loop = true; }

void LoopClosing::updateHashBowDatabase(const LoopClosingInput::Ptr& optical_flow_res) {
  if (optical_flow_res == nullptr) return;

  int64_t t_ns = optical_flow_res->t_ns;
  if (optical_flow_res->keypoints.empty()) return;

  size_t num_cams = optical_flow_res->keypoints.size();

  for (size_t cam_id = 0; cam_id < num_cams; cam_id++) {
    TimeCamId tcid{t_ns, cam_id};
    if (hash_bow_database->has_keyframe(tcid)) continue;

    KeypointsData kd;
    std::vector<KeypointId> keypoint_ids;

    // can optimize here by only computing descriptors for the new kpts
    for (const auto& [kp_id, pos] : optical_flow_res->keypoints[cam_id]) {
      kd.corners.emplace_back(pos.translation().cast<double>());
      keypoint_ids.emplace_back(kp_id);
    }

    // TODO: optimize this
    const basalt::ManagedImage<uint16_t>& man_img_raw = *optical_flow_res->input_images->img_data[cam_id].img;
    const basalt::Image<const uint16_t>& img_raw1 = man_img_raw.SubImage(0, 0, man_img_raw.w, man_img_raw.h);

    computeAngles(img_raw1, kd, true);
    computeDescriptors(img_raw1, kd);

    HashBowVector bow_vector;
    std::vector<FeatureHash> hashes;
    hash_bow_database->compute_bow(kd.corner_descriptors, hashes, bow_vector);

    hash_bow_database->add_to_database(tcid, bow_vector);

    for (size_t i = 0; i < kd.corners.size(); i++) {
      std::bitset<256> descriptor = kd.corner_descriptors[i];
      KeypointId keypoint_id = keypoint_ids[i];

      // if descriptor is all zeros print it
      if (descriptor.none()) {
        std::cout << "Warning: descriptor is all zeros for keypoint id " << keypoint_id << " in tcid " << tcid
                  << std::endl;
      }

      kpt_descriptors[tcid][keypoint_id] = descriptor;

      kpts_positions[tcid][keypoint_id] = kd.corners[i].cast<float>();
    }
  }
}

bool LoopClosing::computeAbsolutePose(const TimeCamId& last_kf_tcid,
                                      const std::vector<Landmark<Scalar>>& last_kf_landmarks,
                                      const std::vector<Landmark<Scalar>>& candidate_kf_landmarks,
                                      const std::vector<std::pair<int, int>>& matches,
                                      std::vector<std::pair<int, int>>& inlier_matches, Sophus::SE3d& absolute_pose) {
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (auto& [left, right] : matches) {
    Eigen::Vector4d tmp;
    if (!calib.intrinsics[last_kf_tcid.cam_id].unproject(last_kf_landmarks[left].obs.at(last_kf_tcid).cast<double>(),
                                                         tmp)) {
      continue;
    }
    Eigen::Vector3d bearing = tmp.head<3>();
    bearing.normalize();
    bearingVectors.push_back(bearing);

    Landmark<Scalar> lm = candidate_kf_landmarks[right];
    TimeCamId host_kf_id = lm.host_kf_id;
    Sophus::SE3f T_w_i = map->getKeyframePose(host_kf_id.frame_id);
    Sophus::SE3f T_i_c = calib.T_i_c[host_kf_id.cam_id].cast<float>();
    Sophus::SE3f T_w_candidate = T_w_i * T_i_c;

    Vec4f pt_c = StereographicParam<Scalar>::unproject(lm.direction);
    pt_c *= 1 / lm.inv_dist;
    pt_c[3] = 1;
    Vec4f pt_w = T_w_candidate * pt_c;
    Eigen::Vector3d point = pt_w.head<3>().template cast<double>();
    points.push_back(point);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);

  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP, !deterministic));

  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = config.loop_closing_pnp_ransac_threshold;
  ransac.max_iterations_ = config.loop_closing_pnp_ransac_iterations;

  ransac.computeModel();

  adapter.sett(ransac.model_coefficients_.topRightCorner<3, 1>());
  adapter.setR(ransac.model_coefficients_.topLeftCorner<3, 3>());

  const opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  ransac.sac_model_->selectWithinDistance(nonlinear_transformation, ransac.threshold_, ransac.inliers_);

  absolute_pose =
      Sophus::SE3d(nonlinear_transformation.topLeftCorner<3, 3>(), nonlinear_transformation.topRightCorner<3, 1>());

  size_t num_inliers = ransac.inliers_.size();
  inlier_matches.reserve(num_inliers);
  for (size_t i = 0; i < num_inliers; i++) {
    inlier_matches.emplace_back(matches[ransac.inliers_[i]]);
  }

  return num_inliers >= static_cast<size_t>(config.loop_closing_pnp_min_inliers);
}

size_t LoopClosing::computeAbsolutePoseMultiCam(
    const FrameId& last_kf_id, const std::vector<FrameId>& candidate_kf_ids,
    const std::vector<Eigen::aligned_unordered_map<KeypointId, Keypoint>>& last_kf_kpts,
    const std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& points_3d,
    const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches,
    std::unordered_map<FrameId, std::vector<KeyframesMatch>>& inlier_matches, Sophus::SE3d& absolute_pose) {
  // OpenGV types
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;
  opengv::absolute_pose::NoncentralAbsoluteAdapter::camCorrespondences_t camCorrespondences;

  // Per-camera offsets and rotations (camera -> viewpoint)
  opengv::translations_t camOffsets;
  opengv::rotations_t camRotations;

  const size_t num_cams = calib.intrinsics.size();
  camOffsets.reserve(num_cams);
  camRotations.reserve(num_cams);

  for (size_t c = 0; c < num_cams; ++c) {
    const auto T_i_c = calib.T_i_c[c].cast<double>();  // camera -> viewpoint
    opengv::translation_t t;
    t << T_i_c.translation().x(), T_i_c.translation().y(), T_i_c.translation().z();
    camOffsets.push_back(t);

    opengv::rotation_t R = T_i_c.rotationMatrix().cast<double>();  // rotation camera -> viewpoint
    camRotations.push_back(R);
  }

  size_t num_matches = 0;
  for (const auto& candidate_kf : candidate_kf_ids) {
    num_matches += matches.at(candidate_kf).size();
  }

  bearingVectors.reserve(num_matches);
  camCorrespondences.reserve(num_matches);
  points.reserve(num_matches);
  std::vector<MatchSource> match_sources;
  match_sources.reserve(num_matches);
  std::vector<std::set<KeypointId>> used_kpt_ids(calib.intrinsics.size());

  for (const auto& candidate_kf : candidate_kf_ids) {
    for (size_t i = 0; i < matches.at(candidate_kf).size(); i++) {
      const auto& match = matches.at(candidate_kf).at(i);

      if (used_kpt_ids[match.source_cam].find(match.source_kpt_id) != used_kpt_ids[match.source_cam].end()) {
        continue;  // skip duplicate keypoints
      }

      if (points_3d.at(TimeCamId{candidate_kf, match.target_cam}).find(match.target_kpt_id) ==
          points_3d.at(TimeCamId{candidate_kf, match.target_cam}).end()) {
        std::cout << "Lost match because of abscence of 3D point: " << match.target_kpt_id << std::endl;
        continue;  // skip if no 3D point found
      }

      TimeCamId tcid_last{last_kf_id, match.source_cam};
      Eigen::Vector4d tmp;
      bool ok = calib.intrinsics[match.source_cam].unproject(  // pixel or normalized observation
          last_kf_kpts[match.source_cam].at(match.source_kpt_id).translation().cast<double>(), tmp);
      if (!ok) {
        continue;  // skip if unproject fails
      }
      Eigen::Vector3d bearing = tmp.head<3>();
      bearing.normalize();

      bearingVectors.push_back(bearing);
      camCorrespondences.push_back(match.source_cam);  // which camera this bearing belongs to

      Eigen::Vector3d world_pt = points_3d.at(TimeCamId{candidate_kf, match.target_cam}).at(match.target_kpt_id);
      points.push_back(world_pt);

      used_kpt_ids[match.source_cam].insert(match.source_kpt_id);
      match_sources.push_back(MatchSource{candidate_kf, i});
    }
  }

  // If we have no correspondences we bail out.
  if (bearingVectors.empty() || points.empty() || bearingVectors.size() != points.size()) {
    return false;
  }

  // Create the non-central adapter.
  opengv::absolute_pose::NoncentralAbsoluteAdapter adapter(bearingVectors, camCorrespondences, points, camOffsets,
                                                           camRotations);

  // RANSAC using the GP3P minimal solver
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

  auto absposeproblem_ptr = std::make_shared<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>(
      adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::GP3P, !deterministic);

  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      config.loop_closing_pnp_ransac_threshold;  // adapt threshold (error metric expects cos-angle or similar)
  ransac.max_iterations_ = config.loop_closing_pnp_ransac_iterations;  // adapt

  bool success = ransac.computeModel();
  if (!success || ransac.inliers_.empty()) {
    return false;
  }

  // Extract pose from best model and refine with nonlinear optimization
  const opengv::transformation_t model = ransac.model_coefficients_;
  adapter.sett(model.topRightCorner<3, 1>());  // translation
  adapter.setR(model.topLeftCorner<3, 3>());   // rotation

  const opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  // Recompute inliers under the nonlinear solution
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation, ransac.threshold_, ransac.inliers_);

  // Convert to Sophus::SE3d
  absolute_pose =
      Sophus::SE3d(nonlinear_transformation.topLeftCorner<3, 3>(), nonlinear_transformation.topRightCorner<3, 1>());

  for (int idx : ransac.inliers_) {  // only calculate this if show_gui
    if (idx < 0 || idx >= static_cast<int>(match_sources.size())) continue;
    const MatchSource& src = match_sources[idx];

    inlier_matches[src.candidate_kf].emplace_back(matches.at(src.candidate_kf).at(src.pair_idx));
  }

  return ransac.inliers_.size();
}

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void LoopClosing::buildOptimizationProblem(const VectorOfConstraints& constraints, MapOfPoses* poses,
                                           ceres::Problem* problem) {
  CHECK(poses != nullptr);
  CHECK(problem != nullptr);
  if (constraints.empty()) return;

  ceres::LossFunction* loss_function = nullptr;
  ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

  for (const auto& constraint : constraints) {
    auto pose_begin_iter = poses->find(constraint.id_begin);
    CHECK(pose_begin_iter != poses->end()) << "Pose with ID: " << constraint.id_begin << " not found.";
    auto pose_end_iter = poses->find(constraint.id_end);
    CHECK(pose_end_iter != poses->end()) << "Pose with ID: " << constraint.id_end << " not found.";

    const Eigen::Matrix<double, 6, 6> sqrt_information = constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

    problem->AddResidualBlock(cost_function, loss_function, pose_begin_iter->second.p.data(),
                              pose_begin_iter->second.q.coeffs().data(), pose_end_iter->second.p.data(),
                              pose_end_iter->second.q.coeffs().data());

    problem->SetManifold(pose_begin_iter->second.q.coeffs().data(), quaternion_manifold);
    problem->SetManifold(pose_end_iter->second.q.coeffs().data(), quaternion_manifold);
  }

  auto pose_to_fix = poses->end();
  --pose_to_fix;

  CHECK(pose_to_fix != poses->end()) << "There are no poses.";
  problem->SetParameterBlockConstant(pose_to_fix->second.p.data());
  problem->SetParameterBlockConstant(pose_to_fix->second.q.coeffs().data());
}

bool LoopClosing::solveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != nullptr);

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  return summary.IsSolutionUsable();
}

}  // namespace basalt
