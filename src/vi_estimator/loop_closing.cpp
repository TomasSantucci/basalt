#include <basalt/vi_estimator/loop_closing.h>
#include <basalt/utils/time_utils.hpp>
#include "basalt/optical_flow/optical_flow.h"
#include "basalt/utils/common_types.h"
#include "basalt/utils/eigen_utils.hpp"
#include "basalt/utils/keypoints.h"
#include "basalt/vi_estimator/landmark_database.h"
#include "basalt/vi_estimator/map_interface.h"

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <Eigen/Core>
#include <chrono>
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

void LoopClosing::initialize() { processing_thread.reset(new std::thread(&LoopClosing::processingLoop, this)); }

void LoopClosing::processingLoop() {
  LoopClosingInput::Ptr loop_closing_input;

  bool notify_lc_finished = false;

  while (true) {
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

    HashBowVector bow_vector;
    indexKeyframe(loop_closing_input, bow_vector);

    loop_closing_input->input_images->addKeyframeTime("lc_keyframe_indexed");

    auto loop_detection_result = std::make_shared<LoopDetectionResult>();
    bool success = detectLoop(loop_closing_input, bow_vector, loop_detection_result);

    loop_closing_input->input_images->addKeyframeTime("lc_loop_detection_finished");

    if (!success) {
      notify_lc_finished = true;
      continue;
    }

    if (out_map_write_queue) out_map_write_queue->push(loop_detection_result);

    LoopClosureDecision::Ptr loop_closure_decision;
    in_lc_dec_res_queue.pop(loop_closure_decision);

    loop_closing_input->input_images->addKeyframeTime("lc_loop_decision_received");

    if (!loop_closure_decision->close_loop) {
      if (out_lc_vis_queue) {
        loop_closing_visualization_data->loop_closed = false;
        out_lc_vis_queue->push(loop_closing_visualization_data);
      }
      continue;
    }

    auto loop_closure_result = std::make_shared<LoopClosureResult>();
    success = closeLoop(loop_closing_input, loop_detection_result, loop_closure_decision, loop_closure_result);

    if (out_map_write_queue && success) out_map_write_queue->push(loop_closure_result);

    notify_lc_finished = !success;

    if (out_lc_vis_queue) {
      loop_closing_visualization_data->loop_closed = success;
      out_lc_vis_queue->push(loop_closing_visualization_data);
    }
  }
}

void LoopClosing::indexKeyframe(const LoopClosingInput::Ptr& loop_closing_input, HashBowVector& bow_vector) {
  if (loop_closing_input == nullptr) return;

  int64_t t_ns = loop_closing_input->t_ns;
  if (loop_closing_input->keypoints.empty()) return;

  for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
    TimeCamId tcid{t_ns, cam_id};
    if (hash_bow_database->has_keyframe(tcid)) continue;

    KeypointsData kd;
    std::vector<KeypointId> keypoint_ids;

    const basalt::ManagedImage<uint16_t>& man_img_raw = *loop_closing_input->input_images->img_data[cam_id].img;
    const basalt::Image<const uint16_t> img_raw = man_img_raw.Reinterpret<const uint16_t>();

    // can optimize here by only computing descriptors for the new kpts
    for (const auto& [kp_id, pos] : loop_closing_input->keypoints[cam_id]) {
      if (!img_raw.InBounds(pos.translation().x(), pos.translation().y(), EDGE_THRESHOLD)) {
        continue;
      }
      kd.corners.emplace_back(pos.translation().cast<double>());
      keypoint_ids.emplace_back(kp_id);
    }

    computeAngles(img_raw, kd, true);
    computeDescriptors(img_raw, kd);

    HashBowVector curr_bow_vector;
    std::vector<FeatureHash> hashes;
    hash_bow_database->compute_bow(kd.corner_descriptors, hashes, curr_bow_vector);
    hash_bow_database->add_to_database(tcid, curr_bow_vector);

    // Use the bow vector of the left camera for later retrieval
    if (cam_id == 0) {
      bow_vector = curr_bow_vector;
    }

    for (size_t i = 0; i < kd.corners.size(); i++) {
      std::bitset<256> descriptor = kd.corner_descriptors[i];
      KeypointId keypoint_id = keypoint_ids[i];

      kpt_descriptors[tcid][keypoint_id] = descriptor;

      // Need to save the position of the keypoints for visualization purposes
      if (out_lc_vis_queue) kpts_positions[tcid][keypoint_id] = kd.corners[i].cast<float>();
    }
  }
}

bool LoopClosing::detectLoop(const LoopClosingInput::Ptr& loop_closing_input, const HashBowVector& bow_vector,
                             LoopDetectionResult::Ptr& loop_detection_result) {
  FrameId curr_kf_id = loop_closing_input->t_ns;

  if (out_lc_vis_queue) {
    loop_closing_visualization_data = std::make_shared<LoopClosingVisualizationData>();
    loop_closing_visualization_data->t_ns = curr_kf_id;
  }

  std::vector<Keypoints>& curr_kf_kpts =
      config.loop_closing_use_all_recent_keypoints ? loop_closing_input->keypoints : loop_closing_input->landmarks;

  std::vector<FrameId> candidate_kfs;
  retrieveCandidates(curr_kf_id, bow_vector, candidate_kfs);

  loop_closing_input->input_images->addKeyframeTime("lc_candidates_retrieved");

  if (candidate_kfs.empty()) return false;

  const int64_t validation_start_ts = std::chrono::steady_clock::now().time_since_epoch().count();
  // Initial matching, island request, island matching, reprojection and pose estimation
  std::array<int64_t, 5> cumulative_times = {0, 0, 0, 0, 0};

  bool loop_detected = false;
  for (const auto& candidate_kf : candidate_kfs) {
    loop_detected =
        validateCandidate(loop_closing_input, candidate_kf, curr_kf_kpts, loop_detection_result, cumulative_times);
    if (loop_detected) break;
  }

  cumulative_times[0] += validation_start_ts;
  for (size_t i = 1; i < cumulative_times.size(); ++i) {
    cumulative_times[i] += cumulative_times[i - 1];
  }

  loop_closing_input->input_images->addKeyframeTime("lc_cumulative_initial_matching_ended", cumulative_times[0]);
  loop_closing_input->input_images->addKeyframeTime("lc_cumulative_island_response_received", cumulative_times[1]);
  loop_closing_input->input_images->addKeyframeTime("lc_cumulative_island_matching_ended", cumulative_times[2]);
  loop_closing_input->input_images->addKeyframeTime("lc_cumulative_reprojection_ended", cumulative_times[3]);
  loop_closing_input->input_images->addKeyframeTime("lc_cumulative_pose_estimation_ended", cumulative_times[4]);
  loop_closing_input->input_images->addKeyframeTime("lc_candidates_validated");
  return loop_detected;
}

void LoopClosing::retrieveCandidates(FrameId curr_kf_id, const HashBowVector& bow_vector,
                                     std::vector<FrameId>& results) {
  results.clear();

  std::vector<std::pair<TimeCamId, double>> query_results;
  int64_t max_t_ns = static_cast<int64_t>(curr_kf_id) - config.loop_closing_frame_time_margin_s * 1e9;
  hash_bow_database->querry_database(bow_vector, config.loop_closing_num_frames_to_match, query_results, &max_t_ns);

  for (const auto& [candidate_kf_tcid, score] : query_results) {
    if (candidate_kf_tcid.frame_id == curr_kf_id) continue;
    if (score < config.loop_closing_frames_to_match_threshold) continue;

    if (std::find(results.begin(), results.end(), candidate_kf_tcid.frame_id) != results.end()) continue;
    results.emplace_back(candidate_kf_tcid.frame_id);
  }
}

bool LoopClosing::validateCandidate(const LoopClosingInput::Ptr& loop_closing_input, const FrameId& candidate_kf,
                                    std::vector<Keypoints>& curr_kf_kpts,
                                    LoopDetectionResult::Ptr& loop_detection_result,
                                    std::array<int64_t, 5>& cumulative_times) {
  FrameId curr_kf_id = loop_closing_input->t_ns;
  std::vector<KeyframesMatch> matches;
  std::vector<KeyframesMatch> inlier_matches;
  size_t initial_matches = 0;
  size_t initial_island_matches = 0;
  size_t initial_inliers = 0;
  size_t reprojection_matches = 0;
  size_t reprojection_inliers = 0;

  Timer t;

  // -- Covisibility check --
  if (checkCovisibility(curr_kf_id, candidate_kf)) return false;

  // -- Initial matching --
  std::vector<std::vector<KeypointId>> candidate_kf_kpts(calib.intrinsics.size());
  for (size_t i = 0; i < calib.intrinsics.size(); i++) {
    TimeCamId candidate_tcid{candidate_kf, i};
    for (const auto& kpt : kpt_descriptors[candidate_tcid]) {
      candidate_kf_kpts[i].emplace_back(kpt.first);
    }
  }
  matchKeyframes(curr_kf_id, candidate_kf, curr_kf_kpts, candidate_kf_kpts, matches);
  consolidateMatches(matches);

  initial_matches = matches.size();
  cumulative_times[0] += t.elapsed_ns();
  t.reset();

  if (initial_matches < static_cast<size_t>(config.loop_closing_min_initial_matches)) return false;

  candidate_kf_kpts.clear();
  matches.clear();

  // -- Query Map Database for the island of the candidate keyframe --
  auto msg = std::make_shared<IslandRequest>();
  msg->keyframe = candidate_kf;
  msg->num_neighbors = config.loop_closing_island_size;
  out_map_read_queue->push(msg);

  auto map_island = std::make_shared<IslandResponse>();
  in_island_res_queue.pop(map_island);
  const Eigen::aligned_map<LandmarkId, Vec3d>& points_3d = map_island->landmarks_3d;
  const std::vector<FrameId>& island_kfs = map_island->keyframes;
  const Eigen::aligned_map<TimeCamId, std::set<LandmarkId>>& keyframe_obs = map_island->keyframe_obs;

  cumulative_times[1] += t.elapsed_ns();
  t.reset();

  // -- Match the current keyframe with all the keyframes in the island --
  for (const auto& neighbor_kf : island_kfs) {
    std::vector<std::vector<KeypointId>> candidate_kf_kpts(calib.intrinsics.size());
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      TimeCamId candidate_tcid{neighbor_kf, i};
      if (keyframe_obs.find(candidate_tcid) == keyframe_obs.end()) {
        continue;
      }
      for (const auto& lm_id : keyframe_obs.at(candidate_tcid)) {
        candidate_kf_kpts[i].emplace_back(lm_id);
      }
    }
    matchKeyframes(curr_kf_id, neighbor_kf, curr_kf_kpts, candidate_kf_kpts, matches);
  }
  consolidateMatches(matches);

  initial_island_matches = matches.size();

  cumulative_times[2] += t.elapsed_ns();
  t.reset();

  if (initial_island_matches < static_cast<size_t>(config.loop_closing_min_island_matches)) return false;

  // -- Geometric verification with PnP --
  Sophus::SE3d curr_kf_estimated_pose;
  initial_inliers = estimateAbsolutePose(curr_kf_kpts, points_3d, matches, inlier_matches, curr_kf_estimated_pose);

  cumulative_times[4] += t.elapsed_ns();
  t.reset();

  if (initial_inliers < static_cast<size_t>(config.loop_closing_pnp_min_inliers) ||
      initial_inliers < initial_island_matches * static_cast<size_t>(config.loop_closing_pnp_inliers_ratio))
    return false;

  // -- Optional reprojection and geometric verification --
  if (config.loop_closing_reproject_landmarks) {
    if (out_lc_vis_queue) {
      loop_closing_visualization_data->reprojected_keypoints.resize(config.loop_closing_cameras_to_reproject.size());
      loop_closing_visualization_data->redetected_keypoints.resize(config.loop_closing_cameras_to_reproject.size());
      loop_closing_visualization_data->rematched_keypoints.resize(config.loop_closing_cameras_to_reproject.size());
    }

    matches = inlier_matches;

    std::vector<std::unordered_set<LandmarkId>> matched_landmarks(calib.intrinsics.size());
    for (const auto& match : matches) {
      matched_landmarks[match.current_kf_cam].insert(match.candidate_kf_kpt_id);
    }

    for (size_t i = 0; i < config.loop_closing_cameras_to_reproject.size(); i++) {
      size_t cam_id = config.loop_closing_cameras_to_reproject[i];

      Eigen::aligned_map<LandmarkId, Eigen::Matrix<double, 3, 1>> points_to_reproject;
      for (const auto& lm_id : keyframe_obs.at(TimeCamId{candidate_kf, cam_id})) {
        if (matched_landmarks[cam_id].find(lm_id) != matched_landmarks[cam_id].end()) continue;

        points_to_reproject[lm_id] = points_3d.at(lm_id);
      }

      Keypoints reprojected_keypoints;
      reprojectLandmarks(curr_kf_estimated_pose, cam_id, points_to_reproject, reprojected_keypoints);

      if (out_lc_vis_queue) {
        loop_closing_visualization_data->reprojected_keypoints[i] = reprojected_keypoints;
      }

      const basalt::ManagedImage<uint16_t>& img = *loop_closing_input->input_images->img_data[cam_id].img;
      for (const auto& kp : reprojected_keypoints) {
        Eigen::aligned_vector<Vec2> detected_kpts;
        Vec2 match_pos;

        std::bitset<256> center_descriptor = kpt_descriptors[TimeCamId{candidate_kf, cam_id}][kp.first];
        bool match_found = searchInWindow(img, kp.second, center_descriptor, detected_kpts, match_pos);

        if (out_lc_vis_queue) {
          loop_closing_visualization_data->redetected_keypoints[i][kp.first] = detected_kpts;
        }

        if (!match_found) continue;

        if (out_lc_vis_queue) loop_closing_visualization_data->rematched_keypoints[i].emplace_back(match_pos);

        curr_kf_kpts[cam_id][kp.first] = Eigen::Translation2f(match_pos);
        matches.emplace_back(KeyframesMatch{cam_id, kp.first, kp.first});
      }
    }

    reprojection_matches = matches.size();

    cumulative_times[3] += t.elapsed_ns();
    t.reset();

    inlier_matches.clear();
    reprojection_inliers =
        estimateAbsolutePose(curr_kf_kpts, points_3d, matches, inlier_matches, curr_kf_estimated_pose);

    cumulative_times[4] += t.elapsed_ns();

    if (reprojection_inliers < static_cast<size_t>(config.loop_closing_reprojected_pnp_min_inliers) ||
        reprojection_inliers < initial_island_matches * static_cast<size_t>(config.loop_closing_pnp_inliers_ratio))
      return false;
  }

  if (config.loop_closing_debug) {
    std::unordered_set<LandmarkId> unique_landmarks;
    std::unordered_set<KeypointId> unique_keypoints;
    for (const auto& match : inlier_matches) {
      unique_landmarks.insert(match.candidate_kf_kpt_id);
      unique_keypoints.insert(match.current_kf_kpt_id);
    }

    std::cout << "[LC]  Loop " << curr_kf_id << " <-> " << candidate_kf << "\n";
    std::cout << "      Matches: " << initial_matches << " -> " << initial_island_matches << " -> " << initial_inliers
              << " inliers ";
    if (config.loop_closing_reproject_landmarks) {
      std::cout << " (reprojected: " << reprojection_matches << " -> " << reprojection_inliers << ")";
    }
    std::cout << "\n";
    std::cout << "      Coverage: " << unique_landmarks.size() << " unique landmarks, " << unique_keypoints.size()
              << " unique keypoints" << std::endl;
  }

  // -- Fill the loop detection result --
  for (const auto& match : inlier_matches) {
    LandmarkId candidate_lm_id = match.candidate_kf_kpt_id;
    LandmarkId curr_lm_id = match.current_kf_kpt_id;

    auto it = loop_detection_result->lm_fusions.find(candidate_lm_id);
    if (it == loop_detection_result->lm_fusions.end()) {
      loop_detection_result->lm_fusions.emplace(curr_lm_id, candidate_lm_id);
    }

    TimeCamId curr_tcid{curr_kf_id, match.current_kf_cam};
    loop_detection_result->curr_kf_obs[curr_tcid][curr_lm_id] =
        curr_kf_kpts[match.current_kf_cam][curr_lm_id].cast<float>();
  }

  loop_detection_result->current_kf_id = curr_kf_id;
  loop_detection_result->candidate_kf_id = candidate_kf;
  loop_detection_result->current_kf_corrected_pose = curr_kf_estimated_pose.cast<float>();
  loop_detection_result->candidates_island = island_kfs;

  if (out_lc_vis_queue) {
    loop_closing_visualization_data->island = loop_detection_result->candidates_island;
    loop_closing_visualization_data->candidate_corrected_pose = loop_detection_result->current_kf_corrected_pose;
    loop_closing_visualization_data->current_keypoints = curr_kf_kpts;
    loop_closing_visualization_data->matches = matches;
    loop_closing_visualization_data->inlier_matches = inlier_matches;

    for (const auto& kf_id : island_kfs) {
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        TimeCamId kf_tcid = TimeCamId{kf_id, i};

        if (keyframe_obs.find(kf_tcid) == keyframe_obs.end()) continue;

        auto& positions = kpts_positions[kf_tcid];

        for (const auto& lm_id : keyframe_obs.at(kf_tcid)) {
          if (positions.find(lm_id) == positions.end()) continue;

          loop_closing_visualization_data->candidate_keypoints[kf_tcid][lm_id] =
              Eigen::Translation2f(positions[lm_id].cast<float>());
        }
      }
    }
  }

  return true;
}

bool LoopClosing::checkCovisibility(const FrameId& kf1_id, const FrameId& kf2_id) {
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

void LoopClosing::matchKeyframes(const FrameId& curr_kf_id, const FrameId& candidate_kf_id,
                                 const std::vector<Keypoints>& curr_kf_kpts,
                                 const std::vector<std::vector<KeypointId>>& candidate_kf_kpts,
                                 std::vector<KeyframesMatch>& matches) {
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
        matches.emplace_back(KeyframesMatch{i, keypoints_vectors[i][match.first], candidate_kf_kpts[j][match.second]});
      }
    }
  }
}

void LoopClosing::consolidateMatches(std::vector<KeyframesMatch>& matches) {
  // Ensure each keypoint, in a camera of the current keyframe, is matched with at most one landmark in the island
  std::vector<std::set<KeypointId>> curr_kpt_ids_per_cam(calib.intrinsics.size());
  std::vector<KeyframesMatch> consolidated_matches;

  for (auto& match : matches) {
    if (curr_kpt_ids_per_cam[match.current_kf_cam].find(match.current_kf_kpt_id) !=
        curr_kpt_ids_per_cam[match.current_kf_cam].end()) {
      continue;
    }

    curr_kpt_ids_per_cam[match.current_kf_cam].insert(match.current_kf_kpt_id);
    consolidated_matches.emplace_back(match);
  }

  matches = std::move(consolidated_matches);
}

size_t LoopClosing::estimateAbsolutePose(const std::vector<Keypoints>& last_kf_kpts,
                                         const Eigen::aligned_map<LandmarkId, Vec3d>& points_3d,
                                         const std::vector<KeyframesMatch>& matches,
                                         std::vector<KeyframesMatch>& inlier_matches, Sophus::SE3d& estimated_pose) {
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;
  opengv::absolute_pose::NoncentralAbsoluteAdapter::camCorrespondences_t camCorrespondences;
  opengv::translations_t camOffsets;
  opengv::rotations_t camRotations;

  const size_t num_cams = calib.intrinsics.size();
  camOffsets.reserve(num_cams);
  camRotations.reserve(num_cams);

  for (size_t c = 0; c < num_cams; ++c) {
    const Sophus::SE3d T_i_c = calib.T_i_c[c];
    opengv::translation_t t;
    t << T_i_c.translation().x(), T_i_c.translation().y(), T_i_c.translation().z();
    camOffsets.push_back(t);

    opengv::rotation_t R = T_i_c.rotationMatrix();
    camRotations.push_back(R);
  }

  size_t num_matches = matches.size();

  bearingVectors.reserve(num_matches);
  camCorrespondences.reserve(num_matches);
  points.reserve(num_matches);

  std::vector<size_t> match_indexes;
  match_indexes.reserve(num_matches);

  for (size_t i = 0; i < matches.size(); i++) {
    const KeyframesMatch& match = matches[i];

    if (points_3d.find(match.candidate_kf_kpt_id) == points_3d.end()) {
      continue;  // Skip if no landmark found
    }

    Eigen::Vector4d tmp;
    bool unproject_success = calib.intrinsics[match.current_kf_cam].unproject(
        last_kf_kpts[match.current_kf_cam].at(match.current_kf_kpt_id).translation().cast<double>(), tmp);

    if (!unproject_success) {
      continue;  // skip if unproject fails
    }

    Eigen::Vector3d bearing = tmp.head<3>();
    bearing.normalize();

    bearingVectors.push_back(bearing);
    camCorrespondences.push_back(match.current_kf_cam);

    Eigen::Vector3d world_pt = points_3d.at(match.candidate_kf_kpt_id);
    points.push_back(world_pt);

    match_indexes.push_back(i);
  }

  if (bearingVectors.empty() || points.empty() || bearingVectors.size() != points.size()) {
    return 0;
  }

  opengv::absolute_pose::NoncentralAbsoluteAdapter adapter(bearingVectors, camCorrespondences, points, camOffsets,
                                                           camRotations);

  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

  auto absposeproblem_ptr = std::make_shared<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>(
      adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::GP3P, !deterministic);

  ransac.sac_model_ = absposeproblem_ptr;
  // ransac.threshold_ = 1.0 - cos(atan(pixel_tolerance/focal_length));
  ransac.threshold_ = config.loop_closing_pnp_ransac_threshold;
  ransac.max_iterations_ = config.loop_closing_pnp_ransac_iterations;

  bool success = ransac.computeModel();
  if (!success || ransac.inliers_.empty()) {
    return 0;
  }

  // Extract pose from best model and refine with nonlinear optimization
  const opengv::transformation_t model = ransac.model_coefficients_;
  adapter.sett(model.topRightCorner<3, 1>());
  adapter.setR(model.topLeftCorner<3, 3>());

  const opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  // Recompute inliers under the nonlinear solution
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation, ransac.threshold_, ransac.inliers_);

  estimated_pose =
      Sophus::SE3d(nonlinear_transformation.topLeftCorner<3, 3>(), nonlinear_transformation.topRightCorner<3, 1>());

  for (int idx : ransac.inliers_) {
    if (idx < 0 || idx >= static_cast<int>(match_indexes.size())) continue;
    const size_t match_idx = match_indexes[idx];

    inlier_matches.emplace_back(matches.at(match_idx));
  }

  return ransac.inliers_.size();
}

void LoopClosing::reprojectLandmarks(const Sophus::SE3d& T_w_i, size_t cam_id,
                                     const Eigen::aligned_map<LandmarkId, Vec3d>& points_3d,
                                     Keypoints& reprojected_keypoints) {
  Sophus::SE3d T_i_c = calib.T_i_c[cam_id];
  Sophus::SE3d T_w_c = T_w_i * T_i_c;
  Sophus::SE3d T_c_w = T_w_c.inverse();

  for (const auto& [lm_id, world_pt] : points_3d) {
    Eigen::Vector3d pt_c = T_c_w * world_pt;

    Eigen::Vector2d uv;
    bool valid = calib.intrinsics[cam_id].project(pt_c, uv);

    if (pt_c.z() <= 0) {
      continue;
    }

    if (!valid) continue;

    // Check if the reprojected point is within the image bounds
    const size_t w = calib.resolution[cam_id].x();
    const size_t h = calib.resolution[cam_id].y();
    if (uv.x() < 0 || uv.x() >= w || uv.y() < 0 || uv.y() >= h) {
      continue;
    }

    reprojected_keypoints[lm_id] = Eigen::Translation2f(uv.cast<float>());
  }
}

bool LoopClosing::searchInWindow(const basalt::ManagedImage<uint16_t>& img, const Keypoint& center,
                                 const std::bitset<256>& center_descriptor, Eigen::aligned_vector<Vec2>& detected_kpts,
                                 Vec2& match_pos) {
  if (!img.InBounds(center.translation().x(), center.translation().y(), config.loop_closing_fast_grid_size / 2.0))
    return false;

  const basalt::Image<const uint16_t>& sub_img_raw =
      img.SubImage(center.translation().x() - (config.loop_closing_fast_grid_size / 2.0),
                   center.translation().y() - (config.loop_closing_fast_grid_size / 2.0),
                   config.loop_closing_fast_grid_size, config.loop_closing_fast_grid_size);

  const basalt::Image<const uint16_t>& img_raw = img.SubImage(0, 0, img.w, img.h);

  KeypointsData kd;
  detectKeypointsFAST(sub_img_raw, kd, config.loop_closing_fast_threshold, config.loop_closing_fast_nonmax_suppression);

  for (size_t i = 0; i < kd.corners.size(); i++) {
    kd.corners[i] += Eigen::Vector2d(center.translation().x() - (config.loop_closing_fast_grid_size / 2.0),
                                     center.translation().y() - (config.loop_closing_fast_grid_size / 2.0));
  }

  // To compute descriptors, we need to filter out the keypoints that are too close to the border of the image
  KeypointsData kd_filtered;
  for (size_t i = 0; i < kd.corners.size(); i++) {
    if (img_raw.InBounds(kd.corners[i].x(), kd.corners[i].y(), EDGE_THRESHOLD)) {
      kd_filtered.corners.emplace_back(kd.corners[i]);
      kd_filtered.corner_responses.emplace_back(kd.corner_responses[i]);
    }
  }
  kd = std::move(kd_filtered);

  for (const auto& kpt : kd.corners) {
    detected_kpts.emplace_back(kpt.cast<float>());
  }

  computeAngles(img_raw, kd, true);
  computeDescriptors(img_raw, kd);

  std::vector<std::pair<int, int>> matches;
  std::vector<std::bitset<256>> target_descriptor_vec = {center_descriptor};
  matchDescriptors(target_descriptor_vec, kd.corner_descriptors, matches,
                   config.loop_closing_redetect_max_hamming_distance,
                   config.loop_closing_redetect_second_best_test_ratio);

  if (matches.size() > 0) {
    match_pos = kd.corners[matches[0].second].cast<float>();
    return true;
  }

  return false;
}

bool LoopClosing::closeLoop(const LoopClosingInput::Ptr& loop_closing_input,
                            const LoopDetectionResult::Ptr& loop_detection_result,
                            const LoopClosureDecision::Ptr& loop_closure_decision,
                            LoopClosureResult::Ptr& loop_closure_result) {
  MapOfPoses map_of_poses;
  VectorOfConstraints constraints;
  buildPoseGraph(loop_detection_result, loop_closure_decision, map_of_poses, constraints);

  if (config.loop_closing_debug) {
    std::cout << "[LC]  Pose graph: " << map_of_poses.size() << " poses, " << constraints.size() << " constraints"
              << std::endl;
  }

  ceres::Problem problem;
  buildCeresProblem(constraints, loop_closure_decision->active_keyframes, &map_of_poses, &problem);

  loop_closing_input->input_images->addKeyframeTime("lc_pgo_problem_built");

  bool success = solveCeresProblem(&problem);

  loop_closing_input->input_images->addKeyframeTime("lc_pgo_problem_solved");

  if (!success) {
    return false;
  }

  applyOptimizedPoses(map_of_poses, *loop_closure_decision->keyframe_poses);

  loop_closing_input->input_images->addKeyframeTime("lc_pgo_poses_applied");

  loop_closure_result->keyframe_poses = loop_closure_decision->keyframe_poses;
  loop_closure_result->loop_detection_result = loop_detection_result;

  return true;
}

void LoopClosing::buildPoseGraph(const LoopDetectionResult::Ptr& loop_detection_result,
                                 const LoopClosureDecision::Ptr& loop_closure_decision, MapOfPoses& map_of_poses,
                                 VectorOfConstraints& constraints) {
  map_of_poses.clear();
  constraints.clear();

  const Eigen::aligned_map<FrameId, Sophus::SE3f>& keyframe_poses = *loop_closure_decision->keyframe_poses;
  const CovisibilityGraph& covisibility_graph = loop_closure_decision->covisibility_graph;
  const FrameId curr_kf_id = loop_detection_result->current_kf_id;
  const Sophus::SE3f curr_kf_corrected_pose = loop_detection_result->current_kf_corrected_pose;

  for (const auto& [kf_id, pose] : keyframe_poses) {
    map_of_poses[kf_id] = Pose3d{pose.translation().cast<double>(), pose.unit_quaternion().cast<double>()};
  }

  // Add the constraints based on the relative poses of the spanning tree
  FrameId spanning_tree_root = covisibility_graph.getRoot();
  for (const auto& [kf_id, T_w_i] : keyframe_poses) {
    if (kf_id == spanning_tree_root) continue;

    FrameId parent_kf_id = covisibility_graph.getParentId(kf_id);
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

  // Add the past loop closure constraints
  for (const auto& [kf_id, loop_closures] : covisibility_graph.getLoopClosures()) {
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
        covisibility_graph.getCovisibleAbove(kf_id, config.loop_closing_pgo_min_covisibility_weight);
    for (const auto& covisible_kf_id : most_covisible_kfs) {
      if (kf_id >= covisible_kf_id) continue;

      if (kf_id != spanning_tree_root) {
        if (covisibility_graph.getParentId(kf_id) == covisible_kf_id) {
          continue;  // already added as part of the spanning tree
        }
      }
      if (covisible_kf_id != spanning_tree_root) {
        if (covisibility_graph.getParentId(covisible_kf_id) == kf_id) {
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

  // Add the current loop closure constraints
  for (const auto& kf_id : loop_detection_result->candidates_island) {
    Sophus::SE3f T_w_i = keyframe_poses.at(kf_id);
    Sophus::SE3f T_w_j = curr_kf_corrected_pose;
    Sophus::SE3f T_i_j = T_w_i.inverse() * T_w_j;
    Constraint3d c;
    c.id_begin = kf_id;
    c.id_end = curr_kf_id;
    Pose3d relative_pose;
    relative_pose.p = T_i_j.translation().cast<double>();
    relative_pose.q = T_i_j.unit_quaternion().cast<double>();
    c.t_be = relative_pose;
    c.information = Eigen::Matrix<double, 6, 6>::Identity();
    constraints.push_back(c);
  }
}

void LoopClosing::buildCeresProblem(const VectorOfConstraints& constraints, const std::set<FrameId>& active_keyframes,
                                    MapOfPoses* poses, ceres::Problem* problem) {
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

  for (const auto& kf_id : active_keyframes) {
    auto pose_iter = poses->find(kf_id);
    if (pose_iter != poses->end()) {
      problem->SetParameterBlockConstant(pose_iter->second.p.data());
      problem->SetParameterBlockConstant(pose_iter->second.q.coeffs().data());
    }
  }
}

bool LoopClosing::solveCeresProblem(ceres::Problem* problem) {
  CHECK(problem != nullptr);

  ceres::Solver::Options options;
  options.max_num_iterations = 20;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  if (config.loop_closing_debug) {
    std::cout << "      Ceres: " << summary.iterations.size() << " iter, " << summary.total_time_in_seconds * 1000.0
              << " ms, cost " << summary.initial_cost << " -> " << summary.final_cost << " ("
              << 100.0 * (summary.initial_cost - summary.final_cost) / summary.initial_cost << "%)" << std::endl;
  }

  return summary.IsSolutionUsable();
}

void LoopClosing::applyOptimizedPoses(const MapOfPoses& map_of_poses,
                                      Eigen::aligned_map<FrameId, Sophus::SE3f>& keyframe_poses) {
  for (const auto& [id, updated_pose] : map_of_poses) {
    if (keyframe_poses.find(id) == keyframe_poses.end()) {
      std::cerr << "Error: Keyframe ID " << id << " not found in keyframe_poses." << std::endl;
      continue;
    }
    Sophus::SE3f& pose = keyframe_poses.at(id);
    pose.translation() = updated_pose.p.cast<float>();
    pose.so3() = Sophus::SO3f(updated_pose.q.cast<float>());
  }
}

}  // namespace basalt
