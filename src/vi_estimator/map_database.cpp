#include <basalt/vi_estimator/map_database.h>
#include "basalt/utils/common_types.h"
#include "basalt/utils/eigen_utils.hpp"
#include "basalt/utils/keypoints.h"
#include "basalt/vi_estimator/landmark_database.h"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

namespace basalt {

MapDatabase::MapDatabase(const VioConfig& config, const Calibration<double>& calib) {
  this->config = config;
  this->calib = calib;
  this->map = LandmarkDatabase<float>("Persistent Map");
  hash_bow_database.reset(new HashBow<256>(config.mapper_bow_num_bits));
}

void MapDatabase::initialize() {
  auto read_func = [&]() {
    while (true) {
      auto keypoints_ptr = std::make_shared<std::vector<KeypointId>>();
      in_covi_req_queue.pop(keypoints_ptr);

      if (keypoints_ptr == nullptr) break;
      std::vector<KeypointId>& keypoints = *keypoints_ptr;

      std::unique_lock<std::mutex> lock(mutex);
      handleCovisibilityReq(keypoints);
    }
  };

  auto write_func = [&]() {
    basalt::MapStamp::Ptr map_stamp;
    while (true) {
      in_map_stamp_queue.pop(map_stamp);

      if (map_stamp == nullptr) {
        map.print();
        if (out_vis_queue) out_vis_queue->push(nullptr);
        if (out_pr_vis_queue) out_pr_vis_queue->push(nullptr);
        break;
      }

      std::unique_lock<std::mutex> lock(mutex);
      map.mergeLMDB(map_stamp->lmdb, true);

      if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_STS) {
        std::set<TimeCamId> kfs_to_compute;
        for (auto const& [kf_id, _] : map_stamp->lmdb->getKeyframeObs()) kfs_to_compute.emplace(kf_id);
        computeSpatialDistributions(kfs_to_compute);
      }

      // Use the last keyframe to detect loops
      TimeCamId maps_last_kf = TimeCamId{map.getLastKeyframe().frame_id, 0};
      std::vector<TimeCamId> similar_kfs;
      std::unordered_map<TimeCamId, SE3> corrected_pose;
      std::unordered_map<TimeCamId, SE3> candidate_pose;
      if (out_pr_vis_queue) {
        place_recognition_visualization_data = std::make_shared<PlaceRecognitionVisualizationData>();
        place_recognition_visualization_data->t_ns = maps_last_kf.frame_id;
      }
      findSimilarKeyframes(maps_last_kf, similar_kfs);

      if (out_pr_vis_queue && place_recognition_visualization_data->similar_kfs.size())
        out_pr_vis_queue->push(place_recognition_visualization_data);

      updateHashBowDatabase(map_stamp->lmdb);

      if (out_vis_queue) {
        map_visual_data = std::make_shared<MapDatabaseVisualizationData>();
        map_visual_data->t_ns = map_stamp->t_ns;
        computeMapVisualData();
        out_vis_queue->push(map_visual_data);
      }
    }
  };

  reading_thread.reset(new std::thread(read_func));
  writing_thread.reset(new std::thread(write_func));
}

void MapDatabase::updateHashBowDatabase(const LandmarkDatabase<Scalar>::Ptr& lmdb) {
  if (lmdb->getKeyframes().empty()) return;

  size_t num_cams = calib.intrinsics.size();

  for (const auto& [frameid, kf] : lmdb->getKeyframes()) {
    for (size_t cam_id = 0; cam_id < num_cams; cam_id++) {
      TimeCamId tcid{frameid, cam_id};
      if (hash_bow_database->has_keyframe(tcid)) continue;

      std::set<LandmarkId> landmarks = lmdb->getKeyframeObs().at(tcid);

      std::vector<std::bitset<256>> descriptors;
      descriptors.reserve(landmarks.size());
      for (const auto& lm_id : landmarks) {
        auto lm = lmdb->getLandmark(lm_id);
        const auto& descriptor = lm.descriptor;
        descriptors.emplace_back(descriptor);
      }

      HashBowVector bow_vector;
      std::vector<FeatureHash> hashes;
      hash_bow_database->compute_bow(descriptors, hashes, bow_vector);

      hash_bow_database->add_to_database(tcid, bow_vector);
    }
  }
}

void MapDatabase::findSimilarKeyframes(const TimeCamId& kf, std::vector<TimeCamId>& similar_kfs) {
  similar_kfs.clear();

  // Obtain the landmarks observed by kf and its descriptors
  std::set<LandmarkId> landmarks = map.getKeyframeObs().at(kf);
  std::vector<std::bitset<256>> descriptors;
  std::vector<Landmark<Scalar>> landmarks_vector;
  descriptors.reserve(landmarks.size());
  landmarks_vector.reserve(landmarks.size());
  for (const auto& lm_id : landmarks) {
    auto lm = map.getLandmark(lm_id);
    const auto& descriptor = lm.descriptor;
    descriptors.emplace_back(descriptor);
    landmarks_vector.emplace_back(lm);
  }

  // Obtain similar keyframes using the hash bow database
  HashBowVector bow_vector;
  std::vector<FeatureHash> hashes;
  hash_bow_database->compute_bow(descriptors, hashes, bow_vector);
  std::vector<std::pair<TimeCamId, double>> results;
  hash_bow_database->querry_database(bow_vector, config.mapper_num_frames_to_match, results);

  if (out_pr_vis_queue) {
    Sophus::SE3f T_w_i = map.getKeyframePose(kf.frame_id);
    Sophus::SE3f T_i_c = calib.T_i_c[kf.cam_id].cast<float>();
    Sophus::SE3f kf_pose = T_w_i * T_i_c;
    place_recognition_visualization_data->keyframe_pose = kf_pose;
  }

  for (const auto& [candidate_kf_tcid, score] : results) {
    if (candidate_kf_tcid.frame_id == kf.frame_id) continue;
    if (score < config.mapper_frames_to_match_threshold) continue;

    // Skip if the candidate keyframe is covisible with the last keyframe
    std::set<LandmarkId> candidate_kf_landmarks = map.getKeyframeObs().at(candidate_kf_tcid);
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
      auto lm = map.getLandmark(lm_id);
      const auto& descriptor = lm.descriptor;
      candidate_kf_descriptors.emplace_back(descriptor);
      candidate_kf_landmarks_vector.emplace_back(lm);
    }

    // Match descriptors between the keyframes
    std::vector<std::pair<int, int>> matches;
    matchDescriptors(descriptors, candidate_kf_descriptors, matches, config.mapper_max_hamming_distance,
                     config.mapper_second_best_test_ratio);

    // Skip if there are less than mapper_min_matches matches
    if (matches.size() < config.mapper_min_matches) continue;

    // Perform geometric verification
    std::vector<std::pair<int, int>> inlier_matches;
    Sophus::SE3d T_last_kf;
    bool valid =
        computeAbsolutePose(kf, landmarks_vector, candidate_kf_landmarks_vector, matches, inlier_matches, T_last_kf);
    if (!valid) continue;

    // The candidate keyframe is valid
    similar_kfs.emplace_back(candidate_kf_tcid);

    if (out_pr_vis_queue) {
      Sophus::SE3f T_w_i = map.getKeyframePose(candidate_kf_tcid.frame_id);
      Sophus::SE3f T_i_c = calib.T_i_c[candidate_kf_tcid.cam_id].cast<float>();
      Sophus::SE3f candidate_pose = T_w_i * T_i_c;
      place_recognition_visualization_data->similar_kfs.emplace_back(candidate_kf_tcid);
      place_recognition_visualization_data->corrected_pose[candidate_kf_tcid] = T_last_kf.cast<float>();
      place_recognition_visualization_data->candidate_pose[candidate_kf_tcid] = candidate_pose;

      for (const auto& [left, right] : inlier_matches) {
        Vec2 left_pos = landmarks_vector[left].obs.at(kf);
        Vec2 right_pos = candidate_kf_landmarks_vector[right].obs.at(candidate_kf_tcid);

        place_recognition_visualization_data->matches[candidate_kf_tcid].emplace_back(
            std::make_pair(left_pos, right_pos));
      }
    }
  }
}

bool MapDatabase::computeAbsolutePose(const TimeCamId& last_kf_tcid,
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
    Sophus::SE3f T_w_i = map.getKeyframePose(host_kf_id.frame_id);
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
          adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));
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
  for (const auto& [tcid_h, target_map] : map.getObservations()) {
    for (const auto& [tcid_t, obs] : target_map) {
      Eigen::Vector3d p1 = map.getKeyframePose(tcid_h.frame_id).template cast<double>().translation();
      Eigen::Vector3d p2 = map.getKeyframePose(tcid_t.frame_id).template cast<double>().translation();
      map_visual_data->covisibility.emplace_back(p1);
      map_visual_data->covisibility.emplace_back(p2);
    }
  }

  // Show observations
  for (const auto& [tcid, obs] : map.getKeyframeObs()) {
    Eigen::Vector3d kf_pos = map.getKeyframePose(tcid.frame_id).template cast<double>().translation();
    auto landmarks_3d = get_landmarks_3d_pos(obs);
    for (const auto& lm_id : obs) {
      map_visual_data->observations[lm_id].emplace_back(kf_pos);
      map_visual_data->observations[lm_id].emplace_back(landmarks_3d[lm_id]);
    }
  }
}

void MapDatabase::handleCovisibilityReq(const std::vector<size_t>& curr_kpts) {
  LandmarkDatabase<Scalar>::Ptr covisible_submap{};

  if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_DEFAULT) {
    covisible_submap = std::make_shared<LandmarkDatabase<Scalar>>("Covisible Submap");
    map.getCovisibilityMap(covisible_submap);
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
    if (static_cast<int>(candidate_kfs.size()) >= config.map_sts_max_size) break;
  }

  map.getSubmap(candidate_kfs, sts_map);
}

}  // namespace basalt
