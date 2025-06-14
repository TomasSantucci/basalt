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
#pragma once

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/utils/imu_types.h>
#include <basalt/utils/eigen_utils.hpp>

namespace basalt {

template <class Scalar_>
struct KeypointObservation {
  using Scalar = Scalar_;

  int kpt_id;
  Eigen::Matrix<Scalar, 2, 1> pos;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// keypoint position defined relative to some frame
template <class Scalar_>
struct Landmark {
  using Scalar = Scalar_;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

  using ObsMap = Eigen::aligned_map<TimeCamId, Vec2>;
  using MapIter = typename ObsMap::iterator;

  // 3D position parameters
  Vec2 direction;
  Scalar inv_dist;

  // Observations
  TimeCamId host_kf_id;
  ObsMap obs;

  LandmarkId id;

  inline void backup() {
    backup_direction = direction;
    backup_inv_dist = inv_dist;
  }

  inline void restore() {
    direction = backup_direction;
    inv_dist = backup_inv_dist;
  }

  friend std::ostream& operator<<(std::ostream& os, const Landmark& lm) {
    os << "Landmark(id=" << lm.id << ", host_kf_id=" << lm.host_kf_id << ")";
    return os;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  Vec2 backup_direction;
  Scalar backup_inv_dist;
};

template <class Scalar_>
class LandmarkDatabase {
 public:
  using Scalar = Scalar_;

  // Non-const
  void addLandmark(LandmarkId lm_id, const Landmark<Scalar>& pos);

  void removeFrame(const FrameId& frame);

  void clear() {
    kpts.clear();
    observations.clear();
  }

  void removeKeyframes(const std::set<FrameId>& kfs_to_marg, const std::set<FrameId>& poses_to_marg,
                       const std::set<FrameId>& states_to_marg_all);

  void addObservation(const TimeCamId& tcid_target, const KeypointObservation<Scalar>& o);

  Landmark<Scalar>& getLandmark(LandmarkId lm_id);

  // Const
  const Landmark<Scalar>& getLandmark(LandmarkId lm_id) const;

  std::vector<TimeCamId> getHostKfs() const;

  std::vector<const Landmark<Scalar>*> getLandmarksForHost(const TimeCamId& tcid) const;

  std::vector<std::pair<LandmarkId, const Landmark<Scalar>*>> getLandmarksForHostWithIds(const TimeCamId& tcid) const;

  const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<LandmarkId>>>& getObservations() const;

  const Eigen::aligned_unordered_map<LandmarkId, Landmark<Scalar>>& getLandmarks() const;

  bool landmarkExists(int lm_id) const;

  size_t numLandmarks() const;

  int numObservations() const;

  int numObservations(LandmarkId lm_id) const;

  void removeLandmark(LandmarkId lm_id);

  void removeObservations(LandmarkId lm_id, const std::set<TimeCamId>& obs);

  inline void backup() {
    for (auto& kv : kpts) kv.second.backup();
  }

  inline void restore() {
    for (auto& kv : kpts) kv.second.restore();
  }

 private:
  using MapIter = typename Eigen::aligned_unordered_map<LandmarkId, Landmark<Scalar>>::iterator;
  MapIter removeLandmarkHelper(MapIter it);
  typename Landmark<Scalar>::MapIter removeLandmarkObservationHelper(MapIter it,
                                                                     typename Landmark<Scalar>::MapIter it2);

  Eigen::aligned_unordered_map<LandmarkId, Landmark<Scalar>> kpts;

  std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<LandmarkId>>> observations;

  static constexpr int min_num_obs = 2;
};

}  // namespace basalt
