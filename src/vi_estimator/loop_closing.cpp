#include <basalt/vi_estimator/loop_closing.h>
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
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <sophus/se3.hpp>
#include <vector>

namespace basalt {

LoopClosing::LoopClosing(const VioConfig& config, const Calibration<double>& calib) {
  this->config = config;
  this->calib = calib;
  hash_bow_database.reset(new HashBow<256>(config.mapper_bow_num_bits));
}

void LoopClosing::initialize() {
  auto proc_func = [&]() {
    OpticalFlowResult::Ptr optical_flow_res;
    int iters_left_to_close_loop = config.loop_closing_frequency;

    bool notify_lc_finished = false;
    while (true) {
      if (notify_lc_finished && sync_lc_finished != nullptr) {
        std::lock_guard<std::mutex> lk(sync_lc_finished->m);
        sync_lc_finished->ready = true;
        sync_lc_finished->cvar.notify_one();
        notify_lc_finished = false;
      }

      in_optical_flow_queue.pop(optical_flow_res);
      if (optical_flow_res == nullptr) {
        if (out_lc_vis_queue) out_lc_vis_queue->push(nullptr);
        break;
      }

      updateHashBowDatabase(optical_flow_res);
      if (sync_hashbow_index != nullptr) {
        std::lock_guard<std::mutex> lk(sync_hashbow_index->m);
        sync_hashbow_index->ready = true;
        sync_hashbow_index->cvar.notify_one();
      }

      if (sync_vio_finished != nullptr) {
        std::unique_lock<std::mutex> lk(sync_vio_finished->m);
        sync_vio_finished->cvar.wait(lk, [&] { return sync_vio_finished->ready; });
        sync_vio_finished->ready = false;
      }

      if (iters_left_to_close_loop > 0) {
        iters_left_to_close_loop--;
      }

      if (config.loop_closing_timestamps.size() > 0) {
        if (config.loop_closing_timestamps.front() < optical_flow_res->t_ns) {
          close_loop = true;
          config.loop_closing_timestamps.erase(config.loop_closing_timestamps.begin());
        }
      }

      if (close_loop || (iters_left_to_close_loop == 0)) {
        close_loop = false;
        iters_left_to_close_loop = config.loop_closing_frequency;

        auto msg = std::make_shared<ReadMapReqMsg>();
        msg->req = true;
        out_map_req_queue->push(msg);
        in_map_res_queue.pop(map);

        if (!map->numKeyframes()) {
          notify_lc_finished = true;
          continue;
        }

        TimeCamId curr_kf_tcid = map->getLastKeyframe();

        if (out_lc_vis_queue) {
          loop_closing_visualization_data = std::make_shared<LoopClosingVisualizationData>();
          loop_closing_visualization_data->t_ns = curr_kf_tcid.frame_id;
        }

        std::vector<TimeCamId> loop_candidates;
        TimeCamId curr_kf_tcid1 = TimeCamId{curr_kf_tcid.frame_id, 0};
        std::vector<Landmark<Scalar>> curr_kf_landmarks;
        std::unordered_map<TimeCamId, std::vector<Landmark<Scalar>>> candidate_kf_landmarks_map;
        std::unordered_map<TimeCamId, std::vector<std::pair<int, int>>> matches;
        detectLoopCandidates(curr_kf_tcid1, loop_candidates, curr_kf_landmarks, candidate_kf_landmarks_map, matches);

        if (loop_candidates.empty()) {
          notify_lc_finished = true;
          continue;
        }

        std::vector<TimeCamId> validated_candidates;
        std::vector<Sophus::SE3f> corrected_poses;
        validateLoopCandidates(curr_kf_tcid1, curr_kf_landmarks, candidate_kf_landmarks_map, loop_candidates, matches,
                               validated_candidates, corrected_poses);

        if (validated_candidates.empty()) {
          notify_lc_finished = true;
          continue;
        }

        // The choice of the best candidate is ad hoc for now
        TimeCamId best_candidate_tcid = validated_candidates[0];
        Sophus::SE3f best_corrected_pose = corrected_poses[0];

        std::vector<Sophus::SE3f> poses;
        std::vector<Sophus::SE3f> relative_poses;

        buildPoseGraph(curr_kf_tcid1, best_candidate_tcid, best_corrected_pose, poses, relative_poses);

        MapOfPoses map_of_poses;
        VectorOfConstraints constraints;
        buildCeresParams(poses, relative_poses, map_of_poses, constraints);

        ceres::Problem problem;
        buildOptimizationProblem(constraints, &map_of_poses, &problem);

        bool success = solveOptimizationProblem(&problem);

        if (!success) {
          notify_lc_finished = true;
          continue;
        }

        restorePosesFromCeres(map_of_poses, poses);

        updateMap(poses, best_candidate_tcid);

        auto map_update_msg = std::make_shared<WriteMapUpdateMsg>();
        map_update_msg->map_update = map;
        if (out_map_update_queue) out_map_update_queue->push(map_update_msg);

        if (out_lc_vis_queue) {
          loop_closing_visualization_data->corrected_loop_poses = poses;
          out_lc_vis_queue->push(loop_closing_visualization_data);
        }
      } else {
        notify_lc_finished = true;
      }
    }
  };

  processing_thread.reset(new std::thread(proc_func));
}

void LoopClosing::detectLoopCandidates(
    const TimeCamId& curr_kf_tcid, std::vector<TimeCamId>& loop_candidates,
    std::vector<Landmark<Scalar>>& curr_kf_landmarks,
    std::unordered_map<TimeCamId, std::vector<Landmark<Scalar>>>& candidate_kf_landmarks_map,
    std::unordered_map<TimeCamId, std::vector<std::pair<int, int>>>& matches) {
  loop_candidates.clear();

  // Obtain the landmarks observed by kf and its descriptors
  std::set<LandmarkId> landmarks = map->getKeyframeObs().at(curr_kf_tcid);
  std::vector<std::bitset<256>> descriptors;
  std::vector<Landmark<Scalar>> landmarks_vector;
  descriptors.reserve(landmarks.size());
  landmarks_vector.reserve(landmarks.size());
  for (const auto& lm_id : landmarks) {
    auto lm = map->getLandmark(lm_id);
    const auto& descriptor = kpt_descriptors[lm_id];
    descriptors.emplace_back(descriptor);
    landmarks_vector.emplace_back(lm);
  }
  curr_kf_landmarks = landmarks_vector;

  // Obtain similar keyframes using the hash bow database
  HashBowVector bow_vector;
  std::vector<FeatureHash> hashes;
  hash_bow_database->compute_bow(descriptors, hashes, bow_vector);
  std::vector<std::pair<TimeCamId, double>> results;
  hash_bow_database->querry_database(bow_vector, config.mapper_num_frames_to_match, results, &curr_kf_tcid.frame_id);

  if (out_lc_vis_queue) {
    Sophus::SE3f T_w_i = map->getKeyframePose(curr_kf_tcid.frame_id);
    loop_closing_visualization_data->keyframe_pose = T_w_i;
  }

  for (const auto& [candidate_kf_tcid, score] : results) {
    if (candidate_kf_tcid.frame_id == curr_kf_tcid.frame_id) continue;
    if (score < config.mapper_frames_to_match_threshold) continue;

    // Skip if the candidate keyframe is covisible with the last keyframe
    std::set<LandmarkId> candidate_kf_landmarks = map->getKeyframeObs().at(candidate_kf_tcid);
    std::vector<LandmarkId> common;
    std::set_intersection(candidate_kf_landmarks.begin(), candidate_kf_landmarks.end(), landmarks.begin(),
                          landmarks.end(), std::back_inserter(common));
    if (!common.empty()) continue;

    // Obtain the landmarks observed by the candidate keyframe and its descriptors
    std::vector<std::bitset<256>> candidate_kf_descriptors;
    std::vector<Landmark<Scalar>> candidate_kf_landmarks_vector;
    candidate_kf_descriptors.reserve(candidate_kf_landmarks.size());
    candidate_kf_landmarks_vector.reserve(candidate_kf_landmarks.size());
    for (const auto& lm_id : candidate_kf_landmarks) {
      auto lm = map->getLandmark(lm_id);
      const auto& descriptor = kpt_descriptors[lm_id];
      candidate_kf_descriptors.emplace_back(descriptor);
      candidate_kf_landmarks_vector.emplace_back(lm);
    }
    candidate_kf_landmarks_map[candidate_kf_tcid] = candidate_kf_landmarks_vector;

    // Match descriptors between the keyframes
    matches[candidate_kf_tcid] = std::vector<std::pair<int, int>>();
    matchDescriptors(descriptors, candidate_kf_descriptors, matches[candidate_kf_tcid],
                     config.mapper_max_hamming_distance, config.mapper_second_best_test_ratio);

    // Skip if there are less than mapper_min_matches matches
    if (matches[candidate_kf_tcid].size() < config.mapper_min_matches) {
      matches.erase(candidate_kf_tcid);
      continue;
    }

    // The candidate keyframe is valid
    loop_candidates.emplace_back(candidate_kf_tcid);
  }
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

      loop_closing_visualization_data->inlier_matches[candidate_kf_tcid] = inlier_matches_vec;
      loop_closing_visualization_data->matches[candidate_kf_tcid] = matches_vec;
      Sophus::SE3f T_w_i = map->getKeyframePose(candidate_kf_tcid.frame_id);
      loop_closing_visualization_data->candidate_pose[candidate_kf_tcid] = T_w_i;
      loop_closing_visualization_data->corrected_pose[candidate_kf_tcid] =
          absolute_pose.cast<float>() * T_i_c.inverse();
      loop_closing_visualization_data->similar_kfs.emplace_back(candidate_kf_tcid);
    }

    validated_candidates.emplace_back(candidate_kf_tcid);
    corrected_poses.emplace_back(absolute_pose.cast<float>() * T_i_c.inverse());
  }
}

void LoopClosing::buildPoseGraph(const TimeCamId& curr_kf_tcid, const TimeCamId& best_candidate_tcid,
                                 const Sophus::SE3f& best_corrected_pose, std::vector<Sophus::SE3f>& poses,
                                 std::vector<Sophus::SE3f>& relative_poses) {
  poses.clear();
  relative_poses.clear();

  const Eigen::aligned_map<FrameId, Sophus::SE3f> keyframe_poses = map->getKeyframes();

  auto itStart = keyframe_poses.lower_bound(best_candidate_tcid.frame_id);
  auto itEnd = keyframe_poses.upper_bound(curr_kf_tcid.frame_id);

  Sophus::SE3f T_w_i = keyframe_poses.at(itStart->first);
  poses.emplace_back(T_w_i);
  itStart++;

  for (; itStart != itEnd; itStart++) {
    Sophus::SE3f T_w_next_i = keyframe_poses.at(itStart->first);
    poses.emplace_back(T_w_next_i);

    Sophus::SE3f T_i_next_i = T_w_i.inverse() * T_w_next_i;

    relative_poses.emplace_back(T_i_next_i);

    T_w_i = T_w_next_i;
  }

  Sophus::SE3f T_corrected_candidate = best_corrected_pose.inverse() * keyframe_poses.at(best_candidate_tcid.frame_id);
  relative_poses.emplace_back(T_corrected_candidate);
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

void LoopClosing::restorePosesFromCeres(const MapOfPoses& map_of_poses, std::vector<Sophus::SE3f>& poses) {
  poses.clear();
  poses.reserve(map_of_poses.size());
  for (const auto& [id, pose] : map_of_poses) {
    poses.emplace_back(pose.q.cast<float>(), pose.p.cast<float>());
  }
}

void LoopClosing::updateMap(const std::vector<Sophus::SE3f>& poses, const TimeCamId& best_candidate_tcid) {
  // transform the poses before itStart to align with the optimized poses
  Sophus::SE3f T_w_correction = map->getKeyframePose(best_candidate_tcid.frame_id).inverse() * poses[0];
  auto itStart2 = map->getKeyframes().lower_bound(best_candidate_tcid.frame_id);
  for (auto it = map->getKeyframes().begin(); it != itStart2; it++) {
    Sophus::SE3f& old_pose = map->getKeyframePose(it->first);
    old_pose = old_pose * T_w_correction;
  }

  // update the poses of the keyframes involved in the loop closure
  auto itStart = map->getKeyframes().lower_bound(best_candidate_tcid.frame_id);
  for (size_t i = 0; i < poses.size(); i++) {
    Sophus::SE3f& old_pose = map->getKeyframePose(itStart->first);
    old_pose = poses[i];
    itStart++;
  }
}

void LoopClosing::triggerLoopClosure() { close_loop = true; }

void LoopClosing::updateHashBowDatabase(const OpticalFlowResult::Ptr& optical_flow_res) {
  if (optical_flow_res == nullptr) return;

  int64_t t_ns = optical_flow_res->t_ns;
  if (optical_flow_res->keypoints.empty()) return;

  size_t num_cams = optical_flow_res->keypoints.size();

  for (size_t cam_id = 0; cam_id < num_cams; cam_id++) {
    TimeCamId tcid{t_ns, cam_id};
    if (hash_bow_database->has_keyframe(tcid)) continue;

    KeypointsData kd;
    std::vector<KeypointId> keypoint_ids;

    for (const auto& [kp_id, pos] : optical_flow_res->keypoints[cam_id]) {
      kd.corners.emplace_back(pos.translation().cast<double>());
      keypoint_ids.emplace_back(kp_id);
    }

    const basalt::ManagedImage<uint16_t>& man_img_raw = *optical_flow_res->input_images->img_data[cam_id].img;
    const basalt::Image<const uint16_t>& img_raw1 = man_img_raw.SubImage(0, 0, man_img_raw.w, man_img_raw.h);

    computeAngles(img_raw1, kd, true);
    computeDescriptors(img_raw1, kd);

    for (size_t i = 0; i < kd.corners.size(); i++) {
      std::bitset<256> descriptor = kd.corner_descriptors[i];
      KeypointId keypoint_id = keypoint_ids[i];

      if (kpt_descriptors.find(keypoint_id) != kpt_descriptors.end()) continue;

      kpt_descriptors[keypoint_id] = descriptor;
    }

    HashBowVector bow_vector;
    std::vector<FeatureHash> hashes;
    hash_bow_database->compute_bow(kd.corner_descriptors, hashes, bow_vector);

    hash_bow_database->add_to_database(tcid, bow_vector);
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
  ransac.threshold_ = config.mapper_pnp_ransac_threshold;
  ransac.max_iterations_ = config.mapper_pnp_ransac_iterations;

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

  return num_inliers >= static_cast<size_t>(config.mapper_pnp_min_inliers);
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
