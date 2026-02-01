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

const int EDGE_THRESHOLD = 19;

LoopClosing::LoopClosing(const VioConfig& config, const Calibration<double>& calib) {
  this->config = config;
  this->calib = calib;
  hash_bow_database.reset(new HashBow<256>(config.loop_closing_bow_num_bits));
}

void LoopClosing::initialize() {
  auto proc_func = [&]() {
    LoopClosingInput::Ptr loop_closing_input;

    bool notify_lc_finished = false;
    std::ofstream dump_loop_detection_result_file("loop_detection_times.csv", std::ios::out);

    if (config.loop_closing_dump_times) {
      lc_time_stats.dumpHeader(dump_loop_detection_result_file);
      lc_time_stats.resetStats();
    }

    while (true) {
      if (config.loop_closing_dump_times && lc_time_stats.current_kf_ts != -1) {
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

      if (config.loop_closing_dump_times) {
        lc_time_stats.addTime(LCTimeStage::Start, true);
        lc_time_stats.current_kf_ts = loop_closing_input->t_ns;
      }

      HashBowVector bow_vector;
      updateHashBowDatabase(loop_closing_input, bow_vector);

      if (config.loop_closing_dump_times) lc_time_stats.addTime(LCTimeStage::HashBowIndex, true);

      TimeCamId best_candidate_tcid;
      Sophus::SE3f best_corrected_pose;
      std::vector<FrameId> best_island;
      std::unordered_map<LandmarkId, LandmarkId> lm_fusions;
      std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Vec2>> curr_lc_obs;
      bool success = runLoopClosure(loop_closing_input, bow_vector, best_candidate_tcid, best_corrected_pose,
                                    best_island, lm_fusions, curr_lc_obs);

      if (!success) {
        if (config.loop_closing_dump_times) lc_time_stats.loop_closed = false;
        notify_lc_finished = true;
        continue;
      }

      if (config.loop_closing_dump_times) lc_time_stats.loop_closed = true;

      auto map_msg = std::make_shared<ReadMapReqMsg>();
      map_msg->frame_id = loop_closing_input->t_ns;
      // TODO@tsantucci: make MDB decide if the loop should be closed
      out_map_req_queue->push(map_msg);
      in_map_res_queue.pop(map_response);

      // get the distance between best_corrected_pose and the current pose
      Sophus::SE3f current_pose = map_response->keyframe_poses->at(loop_closing_input->t_ns).cast<float>();
      float drift_reduced = (current_pose.translation() - best_corrected_pose.translation()).norm();
      if (drift_reduced < config.loop_closing_min_drift_reduction) {
        notify_lc_finished = true;
        continue;
      }

      if (out_lc_vis_queue) {
        loop_closing_visualization_data->current_keyframe_pose =
            map_response->keyframe_poses->at(loop_closing_input->t_ns).cast<float>();
      }

      if (config.close_loops) {
        success = closeLoop(loop_closing_input->t_ns, best_island, best_corrected_pose, lm_fusions, curr_lc_obs);
        if (config.loop_closing_dump_times) lc_time_stats.addTime(LCTimeStage::LoopClosure, true);
        notify_lc_finished = !success;
      } else {
        notify_lc_finished = true;
      }

      if (out_lc_vis_queue) out_lc_vis_queue->push(loop_closing_visualization_data);
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

bool LoopClosing::runLoopClosure(const LoopClosingInput::Ptr& loop_closing_input, const HashBowVector& bow_vector,
                                 TimeCamId& best_candidate_tcid, Sophus::SE3f& best_corrected_pose,
                                 std::vector<FrameId>& best_island,
                                 std::unordered_map<LandmarkId, LandmarkId>& lm_fusions,
                                 std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Vec2>>& curr_lc_obs) {
  FrameId curr_kf_id = loop_closing_input->t_ns;

  if (out_lc_vis_queue) {
    loop_closing_visualization_data = std::make_shared<LoopClosingVisualizationData>();
    loop_closing_visualization_data->t_ns = curr_kf_id;
  }

  std::vector<Keypoints> curr_kf_kpts(calib.intrinsics.size());

  if (config.loop_closing_use_all_recent_keypoints) {
    curr_kf_kpts = loop_closing_input->keypoints;
  } else {
    curr_kf_kpts = loop_closing_input->landmarks;
  }

  std::vector<FrameId> candidate_kfs;
  std::vector<double> candidate_scores;
  query_hashbow_database(curr_kf_id, bow_vector, candidate_kfs, candidate_scores);
  lc_time_stats.addTime(LCTimeStage::HashBowSearch, true);

  if (candidate_kfs.empty()) return false;

  bool loop_found = false;
  // Data needed for visualization after the loop
  std::unordered_map<FrameId, std::vector<KeyframesMatch>> matches;
  std::unordered_map<FrameId, std::vector<KeyframesMatch>> inlier_matches;
  MapIslandResponse::Ptr map_island;
  std::vector<Keypoints> reprojected_keypoints(config.loop_closing_cameras_to_reproject.size());
  std::vector<std::unordered_map<KeypointId, Eigen::aligned_vector<Vec2>>> redetected_keypoints_map(
      config.loop_closing_cameras_to_reproject.size());
  std::vector<Eigen::aligned_vector<Vec2>> rematched_keypoints(config.loop_closing_cameras_to_reproject.size());

  for (const auto& candidate_kf : candidate_kfs) {
    matches.clear();
    inlier_matches.clear();
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

      filter_redundant_matches(matches);

      initial_matches = matches[candidate_kf].size();
      if (initial_matches < config.loop_closing_min_initial_matches) {
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

    map_island = std::make_shared<MapIslandResponse>();
    in_map_3d_points_queue.pop(map_island);
    std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& points_3d = map_island->landmarks_3d_map;
    std::vector<FrameId>& kfs_island = map_island->island_keyframes;

    lc_time_stats.addTime(LCTimeStage::LandmarksRequest, true);

    for (const auto& neighbor_kf : kfs_island) {
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
      match_keyframe(curr_kf_id, curr_kf_kpts, candidate_kf_kpts, neighbor_kf, matches[neighbor_kf]);
    }
    filter_redundant_matches(matches);

    initial_island_matches = 0;
    for (const auto& kf_id : kfs_island) {
      initial_island_matches += matches[kf_id].size();
    }

    if (initial_island_matches < static_cast<size_t>(config.loop_closing_min_matches)) {
      lc_time_stats.addTime(LCTimeStage::IslandMatch, false);
      continue;
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

    std::vector<std::unordered_set<LandmarkId>> matched_landmarks(calib.intrinsics.size());
    for (const auto& [kf, kf_matches] : matches) {
      for (const auto& match : kf_matches) {
        matched_landmarks[match.current_kf_cam].insert(match.candidate_kf_kpt_id);
      }
    }

    for (size_t i = 0; i < config.loop_closing_cameras_to_reproject.size(); i++) {
      size_t cam_id = config.loop_closing_cameras_to_reproject[i];

      if (points_3d.find(TimeCamId{candidate_kf, cam_id}) == points_3d.end()) {
        continue;
      }
      reproject_landmarks(absolute_pose, reprojected_keypoints[i], cam_id,
                          points_3d.at(TimeCamId{candidate_kf, cam_id}), matched_landmarks[cam_id]);
      for (const auto& kp : reprojected_keypoints[i]) {
        Eigen::aligned_vector<Vec2>& redetected_kpts = redetected_keypoints_map[i][kp.first];

        Vec2 best_keypoint_pos;

        std::bitset<256> center_kpt_descriptor = kpt_descriptors[TimeCamId{candidate_kf, cam_id}][kp.first];
        bool match_found = redetect_kpts(kp.second, center_kpt_descriptor, redetected_kpts,
                                         *loop_closing_input->input_images->img_data[cam_id].img, best_keypoint_pos);

        if (match_found) {
          rematched_keypoints[i].emplace_back(best_keypoint_pos);
          if (config.loop_closing_use_rematches) {
            curr_kf_kpts[cam_id][kp.first] = Eigen::Translation2f(best_keypoint_pos);
            matches[candidate_kf].emplace_back(KeyframesMatch{cam_id, cam_id, kp.first, kp.first});
          }
        }
      }
    }
    lc_time_stats.addTime(LCTimeStage::Reprojection, true);

    std::set<LandmarkId> landmarks_matched;
    std::vector<std::set<LandmarkId>> landmarks_matched_per_cam(calib.intrinsics.size());

    if (config.loop_closing_use_rematches) {
      // TODO@tsantucci: this step should be based on the previous inliers, not on all matches
      // recompute the absolute pose with the new matches
      inlier_matches.clear();
      size_t new_num_inliers = computeAbsolutePoseMultiCam(curr_kf_id, kfs_island, curr_kf_kpts, points_3d, matches,
                                                           inlier_matches, absolute_pose);

      lc_time_stats.addTime(LCTimeStage::ReprojectedGeometricVerification, true);

      reprojection_inliers = new_num_inliers;

      for (const auto& [kf_id, inlier_matches_vec] : inlier_matches) {
        for (const auto& match : inlier_matches_vec) {
          landmarks_matched.insert(match.candidate_kf_kpt_id);
          landmarks_matched_per_cam[match.current_kf_cam].insert(match.candidate_kf_kpt_id);
        }
      }

      std::cout << curr_kf_id << " -> " << candidate_kf << ": initial matches = " << initial_matches
                << ", island matches = " << initial_island_matches << ", initial inliers = " << initial_inliers
                << ", reprojection inliers = " << reprojection_inliers << std::endl;

      std::cout << "Landmarks matched: " << landmarks_matched.size() << std::endl;
      std::cout << "Landmarks matched per cam: ";
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        std::cout << "Cam " << i << ": " << landmarks_matched_per_cam[i].size() << " ";
      }
      std::cout << std::endl;
    } else {
      reprojection_inliers = num_inliers;

      std::cout << curr_kf_id << " -> " << candidate_kf << ": initial matches = " << initial_matches
                << ", island matches = " << initial_island_matches << ", initial inliers = " << initial_inliers
                << std::endl;
    }

    // Fuse the landmarks
    for (const auto& [kf_id, inlier_matches_vec] : inlier_matches) {
      for (const auto& match : inlier_matches_vec) {
        LandmarkId candidate_lm_id = match.candidate_kf_kpt_id;
        LandmarkId curr_lm_id = match.current_kf_kpt_id;

        if (lm_fusions.find(candidate_lm_id) == lm_fusions.end()) {
          lm_fusions[curr_lm_id] = candidate_lm_id;
        }

        TimeCamId curr_tcid{curr_kf_id, match.current_kf_cam};
        curr_lc_obs[curr_tcid][curr_lm_id] = curr_kf_kpts[match.current_kf_cam][curr_lm_id].translation().cast<float>();
      }
    }

    best_candidate_tcid = TimeCamId{candidate_kf, 0};
    best_corrected_pose = absolute_pose.cast<float>();
    best_island = kfs_island;
    loop_found = true;
    break;
  }

  if (loop_found) {
    populateVisualizationData(curr_kf_id, best_island, map_island->landmarks_3d_map, matches, inlier_matches,
                              curr_kf_kpts, reprojected_keypoints, redetected_keypoints_map, rematched_keypoints,
                              best_corrected_pose, best_candidate_tcid);
  }

  return loop_found;
}

void LoopClosing::populateVisualizationData(
    FrameId curr_kf_id, const std::vector<FrameId>& kfs_island,
    const std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& points_3d,
    const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches,
    const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& inlier_matches,
    const std::vector<Keypoints>& curr_kf_kpts, const std::vector<Keypoints>& reprojected_keypoints,
    const std::vector<std::unordered_map<KeypointId, Eigen::aligned_vector<Vec2>>>& redetected_keypoints_map,
    const std::vector<Eigen::aligned_vector<Vec2>>& rematched_keypoints, const Sophus::SE3f& best_corrected_pose,
    const TimeCamId& best_candidate_tcid) {
  if (!out_lc_vis_queue) return;

  FrameId candidate_kf = best_candidate_tcid.frame_id;

  loop_closing_visualization_data->island = kfs_island;
  loop_closing_visualization_data->candidate_corrected_pose = best_corrected_pose;
  loop_closing_visualization_data->reprojected_keypoints = reprojected_keypoints;
  loop_closing_visualization_data->redetected_keypoints = redetected_keypoints_map;
  loop_closing_visualization_data->rematched_keypoints = rematched_keypoints;
  loop_closing_visualization_data->current_keypoints = curr_kf_kpts;
  loop_closing_visualization_data->matches = matches;
  loop_closing_visualization_data->inlier_matches = inlier_matches;

  Eigen::aligned_map<LandmarkId, Eigen::Vector3f> landmarks_3d;

  for (const auto& [kf_id, kf_matches] : inlier_matches) {
    for (const auto& match : kf_matches) {
      LandmarkId candidate_lm_id = match.candidate_kf_kpt_id;

      if (landmarks_3d.find(candidate_lm_id) != landmarks_3d.end()) continue;

      TimeCamId candidate_tcid{kf_id, match.candidate_kf_cam};
      if (points_3d.find(candidate_tcid) == points_3d.end()) continue;
      if (points_3d.at(candidate_tcid).find(candidate_lm_id) == points_3d.at(candidate_tcid).end()) continue;

      landmarks_3d[candidate_lm_id] = points_3d.at(candidate_tcid).at(candidate_lm_id).cast<float>();
    }
  }

  for (const auto& [lm_id, pos] : landmarks_3d) {
    loop_closing_visualization_data->landmarks_3d.push_back(pos);
  }

  for (const auto& kf_id : kfs_island) {
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      TimeCamId kf_tcid = TimeCamId{kf_id, i};

      if (points_3d.find(kf_tcid) == points_3d.end()) continue;

      for (const auto& [lm_id, pos] : points_3d.at(kf_tcid)) {
        auto& positions = kpts_positions[kf_tcid];
        if (positions.find(lm_id) == positions.end()) continue;

        loop_closing_visualization_data->candidate_keypoints[kf_tcid][lm_id] =
            Eigen::Translation2f(positions[lm_id].cast<float>());
      }
    }
  }
}

void LoopClosing::filter_redundant_matches(std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches) {
  // TODO@tsantucci: it's better to not need to do this
  std::vector<std::set<KeypointId>> source_kpt_ids_per_cam(calib.intrinsics.size());
  std::vector<KeyframesMatch> filtered_matches;

  for (auto& [frame_id, matches_vec] : matches) {
    for (const auto& match : matches_vec) {
      if (source_kpt_ids_per_cam[match.current_kf_cam].find(match.current_kf_kpt_id) ==
          source_kpt_ids_per_cam[match.current_kf_cam].end()) {
        filtered_matches.emplace_back(match);
        source_kpt_ids_per_cam[match.current_kf_cam].insert(match.current_kf_kpt_id);
      }
    }
    matches_vec = std::move(filtered_matches);
    filtered_matches.clear();
  }
}

void LoopClosing::query_hashbow_database(FrameId curr_kf_id, const HashBowVector& bow_vector,
                                         std::vector<FrameId>& results, std::vector<double>& scores) {
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

bool LoopClosing::redetect_kpts(const Keypoint& center_kpt, std::bitset<256>& center_kpt_descriptor,
                                Eigen::aligned_vector<Vec2>& redetected_kpts, const basalt::ManagedImage<uint16_t>& img,
                                Vec2& best_keypoint_pos) {
  if (!img.InBounds(center_kpt.translation().x(), center_kpt.translation().y(),
                    config.loop_closing_fast_grid_size / 2.0))
    return false;

  const basalt::Image<const uint16_t>& sub_img_raw =
      img.SubImage(center_kpt.translation().x() - (config.loop_closing_fast_grid_size / 2.0),
                   center_kpt.translation().y() - (config.loop_closing_fast_grid_size / 2.0),
                   config.loop_closing_fast_grid_size, config.loop_closing_fast_grid_size);

  const basalt::Image<const uint16_t>& img_raw = img.SubImage(0, 0, img.w, img.h);

  KeypointsData kd;
  detectKeypointsFAST(sub_img_raw, kd, config.loop_closing_fast_threshold, config.loop_closing_fast_nonmax_suppression);

  for (size_t i = 0; i < kd.corners.size(); i++) {
    kd.corners[i] += Eigen::Vector2d(center_kpt.translation().x() - (config.loop_closing_fast_grid_size / 2.0),
                                     center_kpt.translation().y() - (config.loop_closing_fast_grid_size / 2.0));
  }

  KeypointsData kd_filtered;
  for (size_t i = 0; i < kd.corners.size(); i++) {
    if (img_raw.InBounds(kd.corners[i].x(), kd.corners[i].y(), EDGE_THRESHOLD)) {
      kd_filtered.corners.emplace_back(kd.corners[i]);
      kd_filtered.corner_responses.emplace_back(kd.corner_responses[i]);
    }
  }
  kd = kd_filtered;

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

void LoopClosing::reproject_landmarks(const Sophus::SE3d& absolute_pose, Keypoints& reprojected_keypoints,
                                      size_t cam_id, const Eigen::aligned_map<LandmarkId, Vec3d>& points_3d,
                                      const std::unordered_set<LandmarkId>& landmarks_to_skip) {
  // get the landmarks observed in candidate_kf's left cam
  std::set<LandmarkId> candidate_kf_landmarks;

  for (const auto& point : points_3d) {
    if (landmarks_to_skip.find(point.first) != landmarks_to_skip.end()) continue;
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

    reprojected_keypoints[lm_id] = Eigen::Translation2f(uv.cast<float>());
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

void LoopClosing::match_keyframe(const FrameId& curr_kf_id, const std::vector<Keypoints>& curr_kf_kpts,
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

void LoopClosing::triggerLoopClosure() { close_loop = true; }

void LoopClosing::updateHashBowDatabase(const LoopClosingInput::Ptr& optical_flow_res, HashBowVector& bow_vector) {
  if (optical_flow_res == nullptr) return;

  int64_t t_ns = optical_flow_res->t_ns;
  if (optical_flow_res->keypoints.empty()) return;

  size_t num_cams = optical_flow_res->keypoints.size();

  for (size_t cam_id = 0; cam_id < num_cams; cam_id++) {
    TimeCamId tcid{t_ns, cam_id};
    if (hash_bow_database->has_keyframe(tcid)) continue;

    KeypointsData kd;
    std::vector<KeypointId> keypoint_ids;

    // TODO@tsantucci: optimize this
    const basalt::ManagedImage<uint16_t>& man_img_raw = *optical_flow_res->input_images->img_data[cam_id].img;
    const basalt::Image<const uint16_t>& img_raw1 = man_img_raw.SubImage(0, 0, man_img_raw.w, man_img_raw.h);

    // can optimize here by only computing descriptors for the new kpts
    for (const auto& [kp_id, pos] : optical_flow_res->keypoints[cam_id]) {
      if (!img_raw1.InBounds(pos.translation().x(), pos.translation().y(), EDGE_THRESHOLD)) {
        continue;
      }
      kd.corners.emplace_back(pos.translation().cast<double>());
      keypoint_ids.emplace_back(kp_id);
    }

    computeAngles(img_raw1, kd, true);
    computeDescriptors(img_raw1, kd);

    HashBowVector curr_bow_vector;
    std::vector<FeatureHash> hashes;
    hash_bow_database->compute_bow(kd.corner_descriptors, hashes, curr_bow_vector);

    hash_bow_database->add_to_database(tcid, curr_bow_vector);

    if (cam_id == 0) {
      bow_vector = curr_bow_vector;  // use the left camera's bow vector for querying
    }

    for (size_t i = 0; i < kd.corners.size(); i++) {
      std::bitset<256> descriptor = kd.corner_descriptors[i];
      KeypointId keypoint_id = keypoint_ids[i];

      kpt_descriptors[tcid][keypoint_id] = descriptor;

      if (out_lc_vis_queue) kpts_positions[tcid][keypoint_id] = kd.corners[i].cast<float>();
    }
  }
}

size_t LoopClosing::computeAbsolutePoseMultiCam(
    const FrameId& last_kf_id, const std::vector<FrameId>& candidate_kf_ids, const std::vector<Keypoints>& last_kf_kpts,
    const std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Vec3d>>& points_3d,
    const std::unordered_map<FrameId, std::vector<KeyframesMatch>>& matches,
    std::unordered_map<FrameId, std::vector<KeyframesMatch>>& inlier_matches, Sophus::SE3d& absolute_pose) {
  // points_3d should just be like a map of LandmarkId -> Vec3d. no need for TimeCamId
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
  // TODO@tsantucci: no need to keep used_kpt_ids, they are unique
  std::vector<std::set<KeypointId>> used_kpt_ids(calib.intrinsics.size());

  for (const auto& candidate_kf : candidate_kf_ids) {
    for (size_t i = 0; i < matches.at(candidate_kf).size(); i++) {
      const auto& match = matches.at(candidate_kf).at(i);

      if (used_kpt_ids[match.current_kf_cam].find(match.current_kf_kpt_id) !=
          used_kpt_ids[match.current_kf_cam].end()) {
        continue;  // skip duplicate keypoints
      }

      if (points_3d.at(TimeCamId{candidate_kf, match.candidate_kf_cam}).find(match.candidate_kf_kpt_id) ==
          points_3d.at(TimeCamId{candidate_kf, match.candidate_kf_cam}).end()) {
        continue;  // skip if no 3D point found
      }

      TimeCamId tcid_last{last_kf_id, match.current_kf_cam};
      Eigen::Vector4d tmp;
      bool ok = calib.intrinsics[match.current_kf_cam].unproject(  // pixel or normalized observation
          last_kf_kpts[match.current_kf_cam].at(match.current_kf_kpt_id).translation().cast<double>(), tmp);
      if (!ok) {
        continue;  // skip if unproject fails
      }
      Eigen::Vector3d bearing = tmp.head<3>();
      bearing.normalize();

      bearingVectors.push_back(bearing);
      camCorrespondences.push_back(match.current_kf_cam);  // which camera this bearing belongs to

      Eigen::Vector3d world_pt =
          points_3d.at(TimeCamId{candidate_kf, match.candidate_kf_cam}).at(match.candidate_kf_kpt_id);
      points.push_back(world_pt);

      used_kpt_ids[match.current_kf_cam].insert(match.current_kf_kpt_id);
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

  const std::set<FrameId>& not_marg_kfs = map_response->not_marg_kfs;
  for (const auto& kf_id : not_marg_kfs) {
    auto pose_iter = poses->find(kf_id);
    if (pose_iter != poses->end()) {
      problem->SetParameterBlockConstant(pose_iter->second.p.data());
      problem->SetParameterBlockConstant(pose_iter->second.q.coeffs().data());
    }
  }
}

bool LoopClosing::solveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != nullptr);

  ceres::Solver::Options options;
  // TODO@tsantucci: lower this ASAP!
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  return summary.IsSolutionUsable();
}

}  // namespace basalt
