/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <basalt/vi_estimator/landmark_database.h>

namespace basalt {

template <class Scalar_>
LandmarkDatabase<Scalar_>::LandmarkDatabase(std::string name) : debug_name(name) {
  keyframe_poses = std::make_shared<Eigen::aligned_map<FrameId, SE3>>();
}

template <class Scalar_>
LandmarkDatabase<Scalar_>::LandmarkDatabase(const LandmarkDatabase& other)
    : kpts(other.kpts),
      observations(other.observations),
      keyframe_idx(other.keyframe_idx),
      keyframe_obs(other.keyframe_obs),
      debug_name(other.debug_name) {
  if (other.keyframe_poses) {
    keyframe_poses = std::make_shared<Eigen::aligned_map<FrameId, SE3>>(*other.keyframe_poses);
  } else {
    keyframe_poses = nullptr;
  }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::addLandmark(LandmarkId lm_id, const Landmark<Scalar>& pos) {
  auto& kpt = kpts[lm_id];
  kpt.direction = pos.direction;
  kpt.inv_dist = pos.inv_dist;
  kpt.host_kf_id = pos.host_kf_id;
  kpt.id = lm_id;
  keyframe_obs[pos.host_kf_id].insert(lm_id);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeFrame(const FrameId& frame) {
  for (auto it = kpts.begin(); it != kpts.end();) {
    for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
      if (it2->first.frame_id == frame) it2 = removeLandmarkObservationHelper(it, it2);
      else it2++;
    }

    if (it->second.obs.size() < min_num_obs) {
      it = removeLandmarkHelper(it);
    } else {
      ++it;
    }
  }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeKeyframes(const std::set<FrameId>& kfs_to_marg,
                                                const std::set<FrameId>& poses_to_marg,
                                                const std::set<FrameId>& states_to_marg_all) {
  for (auto it = kpts.begin(); it != kpts.end();) {
    if (kfs_to_marg.count(it->second.host_kf_id.frame_id) > 0) {
      it = removeLandmarkHelper(it);
    } else {
      for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
        FrameId fid = it2->first.frame_id;
        if (poses_to_marg.count(fid) > 0 || states_to_marg_all.count(fid) > 0 || kfs_to_marg.count(fid) > 0)
          it2 = removeLandmarkObservationHelper(it, it2);
        else it2++;
      }

      if (it->second.obs.size() < min_num_obs) {
        it = removeLandmarkHelper(it);
      } else {
        ++it;
      }
    }
  }
  for (const auto& kf_id : kfs_to_marg) {
    keyframe_idx.erase(kf_id);
    keyframe_poses->erase(kf_id);
  }

  std::vector<TimeCamId> tcid_to_rm;
  for (const auto& [tcid, _] : keyframe_obs)
    if (kfs_to_marg.count(tcid.frame_id) > 0) tcid_to_rm.push_back(tcid);

  for (const auto& tcid : tcid_to_rm) keyframe_obs.erase(tcid);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeKeyframe(FrameId kf_id, int num_cams,
                                               std::vector<Landmark<Scalar>>& removed_landmarks) {
  for (size_t cam_id = 0; cam_id < static_cast<size_t>(num_cams); cam_id++) {
    TimeCamId tcid{kf_id, cam_id};

    auto obs_it = keyframe_obs.find(tcid);
    if (obs_it == keyframe_obs.end()) continue;

    // Copy the set of observed landmarks to avoid modifying it while iterating
    const std::set<LandmarkId> lm_ids_copy = obs_it->second;
    for (const auto& lm_id : lm_ids_copy) {
      auto lm_it = kpts.find(lm_id);
      if (lm_it == kpts.end()) continue;

      Landmark<Scalar>& lm = lm_it->second;

      if (lm.host_kf_id.frame_id == kf_id) {
        removed_landmarks.push_back(lm);
        removeLandmarkHelper(lm_it);
        continue;
      }

      auto obs_it2 = lm.obs.find(tcid);
      if (obs_it2 == lm.obs.end()) continue;

      removeLandmarkObservationHelper(lm_it, obs_it2);
      if (lm.obs.size() < min_num_obs) {
        removed_landmarks.push_back(lm);
        removeLandmarkHelper(lm_it);
      }
    }
  }

  // Remove the keyframe and its pose
  keyframe_idx.erase(kf_id);
  keyframe_poses->erase(kf_id);

  // Remove the keyframe observations
  for (size_t cam_id = 0; cam_id < static_cast<size_t>(num_cams); cam_id++) {
    TimeCamId tcid{kf_id, static_cast<CamId>(cam_id)};
    keyframe_obs.erase(tcid);
  }
}

template <class Scalar_>
std::vector<TimeCamId> LandmarkDatabase<Scalar_>::getHostKfs() const {
  std::vector<TimeCamId> res;

  res.reserve(observations.size());
  for (const auto& [h, _] : observations) res.emplace_back(h);

  return res;
}

template <class Scalar_>
std::vector<const Landmark<Scalar_>*> LandmarkDatabase<Scalar_>::getLandmarksForHost(const TimeCamId& tcid) const {
  std::vector<const Landmark<Scalar>*> res;

  for (const auto& [k, obs] : observations.at(tcid))
    for (const auto& v : obs) res.emplace_back(&kpts.at(v));

  return res;
}

template <class Scalar_>
std::vector<std::pair<LandmarkId, const Landmark<Scalar_>*>> LandmarkDatabase<Scalar_>::getLandmarksForHostWithIds(
    const TimeCamId& tcid) const {
  std::vector<std::pair<LandmarkId, const Landmark<Scalar_>*>> res;
  std::unordered_set<LandmarkId> lm_ids;

  for (const auto& [k, obs] : observations.at(tcid))
    for (const auto& v : obs) lm_ids.insert(v);

  res.reserve(lm_ids.size());
  for (const auto& lm_id : lm_ids) res.emplace_back(lm_id, &kpts.at(lm_id));

  return res;
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::addObservation(const TimeCamId& tcid_target, const KeypointObservation<Scalar>& o) {
  auto it = kpts.find(o.kpt_id);
  BASALT_ASSERT(it != kpts.end());

  it->second.obs[tcid_target] = o.pos;

  observations[it->second.host_kf_id][tcid_target].insert(it->first);

  keyframe_obs[tcid_target].insert(it->first);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::addKeyframe(int64_t kf_id, size_t idx, const SE3& pos) {
  keyframe_idx[kf_id] = idx;
  (*keyframe_poses)[kf_id] = pos;
}

template <class Scalar_>
Sophus::SE3<Scalar_>& LandmarkDatabase<Scalar_>::getKeyframePose(int64_t kf_id) {
  return keyframe_poses->at(kf_id);
}

template <class Scalar_>
size_t& LandmarkDatabase<Scalar_>::getKeyframeIndex(FrameId kf_id) {
  return keyframe_idx.at(kf_id);
}

template <class Scalar_>
TimeCamId LandmarkDatabase<Scalar_>::getLastKeyframe() {
  if (keyframe_obs.empty()) std::cout << "There is no keyframes yet.";

  // Return the last added keyframe
  return keyframe_obs.rbegin()->first;
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::getCovisibilityMap(LandmarkDatabase<Scalar>::Ptr submap) {
  submap->clear();
  if (!keyframe_obs.empty()) {
    TimeCamId last_tcid = getLastKeyframe();
    std::set<TimeCamId> tcids;
    for (const auto& lm_id : keyframe_obs[last_tcid]) {
      auto lm = getLandmark(lm_id);
      for (const auto& [tcid_target, _] : lm.obs) tcids.emplace(tcid_target);
    }
    getSubmap(tcids, submap);
  }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::getSubmap(std::set<TimeCamId> tcids, LandmarkDatabase<Scalar>::Ptr submap) {
  submap->clear();
  for (const auto& tcid : tcids) {
    auto kf_id = tcid.frame_id;
    if (keyframeExists(kf_id)) submap->addKeyframe(kf_id, getKeyframeIndex(kf_id), getKeyframePose(kf_id));

    for (const auto& lm_id : keyframe_obs[tcid]) {
      auto lm = getLandmark(lm_id);

      // NOTE: Avoid adding landmarks whose host keyframe is not in the provided set 'tcids'. This prevents drift issues
      // in Basalt.
      if (tcids.count(lm.host_kf_id) == 0) continue;

      submap->addLandmark(lm_id, lm);

      for (const auto& [tcid_target, pos] : lm.obs) {
        if (tcids.count(tcid_target) > 0) {
          KeypointObservation<Scalar> kobs;
          kobs.kpt_id = lm_id;
          kobs.pos = pos;
          submap->addObservation(tcid_target, kobs);
        }
      }
    }
  }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::mergeLMDB(LandmarkDatabase<Scalar>::Ptr lmdb, bool override) {
  // Add keyframes
  for (const auto& [kf_id, pose] : lmdb->getKeyframes()) {
    if (!override && keyframeExists(kf_id)) continue;  // Skip if the keyframe already exists
    auto idx = lmdb->getKeyframeIndex(kf_id);
    addKeyframe(kf_id, idx, pose);
  }

  // Add Landmarks
  for (const auto& [lm_id, lm] : lmdb->getLandmarks()) {
    // If the landmark will be overridden and it already exists with a different host keyframe,
    // merge observations first
    if (override && landmarkExists(lm_id)) {
      const auto& existing_lm = getLandmark(lm_id);
      if (existing_lm.host_kf_id.frame_id != lm.host_kf_id.frame_id) { mergeObservations(existing_lm, lm); }
    }

    if (override || !landmarkExists(lm_id)) addLandmark(lm_id, lm);

    // Add Observations
    for (const auto& [tcid_target, pos] : lm.obs) {
      if (!keyframeExists(tcid_target.frame_id) && !lmdb->keyframeExists(tcid_target.frame_id))
        continue;  // Basalt adds observations to the LMDB before the frames are keyframes..

      if (!override && observationExists(tcid_target, lm_id)) continue;

      KeypointObservation<Scalar> kobs;
      kobs.kpt_id = lm_id;
      kobs.pos = pos;
      addObservation(tcid_target, kobs);
    }

    // Remove the landmark if it has less than min_num_obs observations between both lmdb.
    // This check must be here because not all the observations are added the current lmdb
    if (numObservations(lm_id) < min_num_obs) {
      removeLandmark(lm_id);
      continue;
    }
  }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::mergeObservations(const Landmark<Scalar_>& existing_lm,
                                                  const Landmark<Scalar_>& incoming_lm) {
  for (const auto& [tcid, _] : existing_lm.obs) {
    observations[existing_lm.host_kf_id][tcid].erase(existing_lm.id);
    if (observations[existing_lm.host_kf_id][tcid].empty()) {
      observations[existing_lm.host_kf_id].erase(tcid);
      if (observations[existing_lm.host_kf_id].empty()) observations.erase(existing_lm.host_kf_id);
    }

    observations[incoming_lm.host_kf_id][tcid].insert(incoming_lm.id);
  }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::mergeLandmarks(const LandmarkId& from_lm_id, const LandmarkId& to_lm_id) {
  if (from_lm_id == to_lm_id) return;

  // Get the landmarks
  auto& from_lm = getLandmark(from_lm_id);
  auto& to_lm = getLandmark(to_lm_id);

  // Merge observations
  for (const auto& [tcid, pos] : from_lm.obs) {
    // If the observation already exists in to_lm, skip it
    if (to_lm.obs.count(tcid) > 0) continue;

    // Add the observation to to_lm
    to_lm.obs[tcid] = pos;

    // Update the observations map
    observations[to_lm.host_kf_id][tcid].insert(to_lm_id);
    keyframe_obs[tcid].insert(to_lm_id);
  }

  // Remove the from_lm landmark
  removeLandmark(from_lm_id);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeReferencesToCulledKeyframes(const LandmarkDatabase<Scalar>& persistent_map,
                                                                  FrameId last_persistent_kf) {
  /* This removes from the current LMDB all the references to keyframes that do not exist in the persistent map.
     If a keyframe is not in the persistent map, but is more recent than the last keyframe in the persistent map,
     it is not removed, since it might be added later to the persistent map during the merge.

     For each landmark in the current LMDB stamp:
     1) if the host keyframe does not exist in the persistent map, remove the landmark from the current LMDB
     2) otherwise, remove all the observations that reference inexistent keyframes in the persistent map
  */
  // last_persistent_kf = persistent_map.getKeyframes().empty() ? -1 : persistent_map.getKeyframes().rbegin()->first;

  // TODO@tsantucci: remove print statements and replace with proper logging
  for (auto it = kpts.begin(); it != kpts.end();) {
    if (!persistent_map.keyframeExists(it->second.host_kf_id.frame_id) &&
        it->second.host_kf_id.frame_id <= last_persistent_kf) {
      std::cout << "[LMDB] Removing landmark " << it->first << " because its host keyframe "
                << it->second.host_kf_id.frame_id
                << " does not exist in the persistent map and is older than the last persistent keyframe "
                << last_persistent_kf << std::endl;
      it = removeLandmarkHelper(it);
      continue;
    }

    for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
      if (!persistent_map.keyframeExists(it2->first.frame_id) && it2->first.frame_id <= last_persistent_kf) {
        std::cout << "[LMDB] Removing observation of landmark " << it->first << " in keyframe " << it2->first.frame_id
                  << " because the keyframe does not exist in the persistent map and is older than the last persistent "
                     "keyframe "
                  << last_persistent_kf << std::endl;
        it2 = removeLandmarkObservationHelper(it, it2);
      } else {
        it2++;
      }
    }
    it++;
  }
}

template <class Scalar_>
Landmark<Scalar_>& LandmarkDatabase<Scalar_>::getLandmark(LandmarkId lm_id) {
  return kpts.at(lm_id);
}

template <class Scalar_>
const Landmark<Scalar_>& LandmarkDatabase<Scalar_>::getLandmark(LandmarkId lm_id) const {
  return kpts.at(lm_id);
}

template <class Scalar_>
const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<LandmarkId>>>&
LandmarkDatabase<Scalar_>::getObservations() const {
  return observations;
}

template <class Scalar_>
const Eigen::aligned_unordered_map<LandmarkId, Landmark<Scalar_>>& LandmarkDatabase<Scalar_>::getLandmarks() const {
  return kpts;
}

template <class Scalar_>
const Eigen::aligned_map<FrameId, Sophus::SE3<Scalar_>>& LandmarkDatabase<Scalar_>::getKeyframes() const {
  if (keyframe_poses->empty()) {
    static const Eigen::aligned_map<FrameId, Sophus::SE3<Scalar_>> empty_map{};
    return empty_map;
  }
  return *keyframe_poses;
}

template <class Scalar_>
const Eigen::aligned_map<TimeCamId, std::set<LandmarkId>>& LandmarkDatabase<Scalar_>::getKeyframeObs() const {
  return keyframe_obs;
}

template <class Scalar_>
float LandmarkDatabase<Scalar_>::getKeyframeRedundancyScore(FrameId kf_id, int num_cams,
                                                            bool exclude_hosted_lms) const {
  std::unordered_set<LandmarkId> observed_lms;

  for (int cam_id = 0; cam_id < num_cams; cam_id++) {
    TimeCamId tcid(kf_id, cam_id);
    if (keyframe_obs.count(tcid) > 0) {
      for (const auto& lm_id : keyframe_obs.at(tcid)) observed_lms.insert(lm_id);
    }
  }

  // A landmark is considered redundant if the keyframe is not its host,
  // and it is observed by more than 2 other keyframes
  int redundant_lms = 0;
  for (const auto& lm_id : observed_lms) {
    const auto& lm = getLandmark(lm_id);

    if (exclude_hosted_lms && lm.host_kf_id.frame_id == kf_id) continue;

    std::unordered_set<FrameId> observing_kfs;
    for (const auto& [tcid, _] : lm.obs) {
      observing_kfs.insert(tcid.frame_id);
      if (observing_kfs.size() > 3) {
        redundant_lms++;
        break;
      }
    }
  }

  int total_lms_observed = observed_lms.size();

  return total_lms_observed > 0 ? static_cast<float>(redundant_lms) / static_cast<float>(total_lms_observed) : 0.0f;
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::mergeKeyframesPoses(
    std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3<Scalar_>>> loop_kfs_poses) {
  // This function merges the keyframe poses of loop_kfs_poses into the current LMDB.
  // It adds the new poses present in keyframe_poses but not in loop_kfs_poses
  // It removes the old poses present in loop_kfs_poses but not in keyframe_poses (culled while the loop-closing thread
  // was running)

  FrameId last_kf_pose = loop_kfs_poses->rbegin()->first;

  // Remove the culled keyframe poses from loop_kfs_poses
  for (auto it = loop_kfs_poses->begin(); it != loop_kfs_poses->end();) {
    if (keyframe_poses->count(it->first) == 0) {
      it = loop_kfs_poses->erase(it);
    } else {
      it++;
    }
  }

  // Add the new keyframe poses from keyframe_poses to loop_kfs_poses
  for (auto it = keyframe_poses->upper_bound(last_kf_pose); it != keyframe_poses->end(); ++it) {
    (*loop_kfs_poses)[it->first] = it->second;
  }

  keyframe_poses = loop_kfs_poses;
}

template <class Scalar_>
bool LandmarkDatabase<Scalar_>::landmarkExists(int lm_id) const {
  return kpts.count(lm_id) > 0;
}

template <class Scalar_>
bool LandmarkDatabase<Scalar_>::keyframeExists(FrameId kf_id) const {
  return keyframe_poses->count(kf_id) > 0;
}

template <class Scalar_>
bool LandmarkDatabase<Scalar_>::observationExists(TimeCamId target_tcid, LandmarkId lm_id) {
  // Check if the observation (host, target) -> lm exists
  if (!landmarkExists(lm_id)) return false;
  auto lm = getLandmark(lm_id);

  auto it1 = observations.find(lm.host_kf_id);
  if (it1 == observations.end()) return false;

  auto it2 = it1->second.find(target_tcid);
  if (it2 == it1->second.end()) return false;

  return it2->second.count(lm_id) > 0;
}

template <class Scalar_>
size_t LandmarkDatabase<Scalar_>::numLandmarks() const {
  return kpts.size();
}

template <class Scalar_>
int LandmarkDatabase<Scalar_>::numObservations() const {
  int num_observations = 0;

  for (const auto& [_, val_map] : observations) {
    for (const auto& [_, val] : val_map) { num_observations += val.size(); }
  }

  return num_observations;
}

template <class Scalar_>
int LandmarkDatabase<Scalar_>::numObservations(LandmarkId lm_id) const {
  if (kpts.count(lm_id) == 0) return 0;
  return kpts.at(lm_id).obs.size();
}

template <class Scalar_>
int LandmarkDatabase<Scalar_>::numKeyframes() const {
  return keyframe_poses->size();
}

template <class Scalar_>
typename LandmarkDatabase<Scalar_>::MapIter LandmarkDatabase<Scalar_>::removeLandmarkHelper(
    LandmarkDatabase<Scalar>::MapIter it) {
  auto host_it = observations.find(it->second.host_kf_id);

  if (host_it != observations.end()) {
    for (const auto& [k, v] : it->second.obs) {
      auto target_it = host_it->second.find(k);
      target_it->second.erase(it->first);

      if (target_it->second.empty()) host_it->second.erase(target_it);
    }

    if (host_it->second.empty()) observations.erase(host_it);
  }

  for (const auto& [tcid, _] : it->second.obs) {
    auto obs_it = keyframe_obs.find(tcid);
    if (obs_it != keyframe_obs.end()) {
      obs_it->second.erase(it->first);
      if (obs_it->second.empty()) { keyframe_obs.erase(obs_it); }
    }
  }

  return kpts.erase(it);
}

template <class Scalar_>
typename Landmark<Scalar_>::MapIter LandmarkDatabase<Scalar_>::removeLandmarkObservationHelper(
    LandmarkDatabase<Scalar>::MapIter it, typename Landmark<Scalar>::MapIter it2) {
  auto host_it = observations.find(it->second.host_kf_id);
  auto target_it = host_it->second.find(it2->first);
  target_it->second.erase(it->first);

  if (target_it->second.empty()) host_it->second.erase(target_it);
  if (host_it->second.empty()) observations.erase(host_it);

  auto kf_obs_it = keyframe_obs.find(it2->first);
  if (kf_obs_it != keyframe_obs.end()) {
    kf_obs_it->second.erase(it->first);
    if (kf_obs_it->second.empty()) keyframe_obs.erase(kf_obs_it);
  }

  return it->second.obs.erase(it2);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeLandmark(LandmarkId lm_id) {
  auto it = kpts.find(lm_id);
  if (it != kpts.end()) removeLandmarkHelper(it);
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::removeObservations(LandmarkId lm_id, const std::set<TimeCamId>& obs) {
  auto it = kpts.find(lm_id);
  BASALT_ASSERT(it != kpts.end());

  for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
    if (obs.count(it2->first) > 0) {
      it2 = removeLandmarkObservationHelper(it, it2);
    } else it2++;
  }

  if (it->second.obs.size() < min_num_obs) { removeLandmarkHelper(it); }
}

template <class Scalar_>
void LandmarkDatabase<Scalar_>::print(bool show_ids) {
  // Print database header
  std::cout << "---------------------------------------------------" << std::endl;
  std::cout << "| " << std::setw(48) << std::left << debug_name << "|" << std::endl;
  std::cout << "---------------------------------------------------" << std::endl;

  // Print keyframes
  if (show_ids) {
    std::cout << "| Keyframes: ";
    for (const auto& [_, idx] : keyframe_idx) std::cout << idx << ", ";
    std::cout << std::endl;
  } else std::cout << "| Keyframes count: " << numKeyframes() << std::endl;

  // Print landmarks
  if (show_ids) {
    std::cout << "| Landmarks: ";
    for (const auto& [lm_id, _] : kpts) std::cout << lm_id << ", ";
    std::cout << std::endl;
  } else std::cout << "| Landmarks count: " << numLandmarks() << std::endl;

  // Print observations
  if (show_ids) {
    std::cout << "| Observations: " << std::endl;
    for (const auto& [tcid, landmarks] : keyframe_obs) {
      if (!keyframeExists(tcid.frame_id)) continue;  // tcid is not a timestamp of a keyframe
      auto kf_idx = getKeyframeIndex(tcid.frame_id);
      std::cout << " - Keyframe " << kf_idx << "." << tcid.cam_id << ": ";
      for (const auto& lm_id : landmarks) std::cout << lm_id << ", ";
      std::cout << std::endl;
    }
  } else std::cout << "| Observations count: " << numObservations() << std::endl;

  // Print database footer
  std::cout << "---------------------------------------------------" << std::endl;
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

// Note: double specialization is unconditional, b/c NfrMapper depends on it.
// #ifdef BASALT_INSTANTIATIONS_DOUBLE
template class LandmarkDatabase<double>;
// #endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class LandmarkDatabase<float>;
#endif

}  // namespace basalt
