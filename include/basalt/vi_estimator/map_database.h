#pragma once

#include <basalt/hash_bow/hash_bow.h>
#include <basalt/utils/sync_utils.h>
#include <basalt/vi_estimator/covisibility_graph.h>
#include <basalt/vi_estimator/landmark_database.h>
#include <basalt/vi_estimator/map_interface.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <thread>

namespace basalt {

// TODO: Make an abstract struct SpatialDistribution
template <class Scalar_>
struct SpatialDistributionCube {
 public:
  using Scalar = Scalar_;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  SpatialDistributionCube() = default;

  SpatialDistributionCube(std::vector<Vec3> points) {
    Vec3 mean = Vec3::Zero();
    Vec3 variance = Vec3::Zero();

    for (const auto& point : points) {
      mean += point;
    }
    mean /= points.size();

    for (const auto& point : points) {
      Vec3 diff = point - mean;
      variance += (diff.array() * diff.array()).matrix();
    }
    variance /= points.size();
    Cx << mean.x() - sqrt(variance.x()), mean.x() + sqrt(variance.x());
    Cy << mean.y() - sqrt(variance.y()), mean.y() + sqrt(variance.y());
    Cz << mean.z() - sqrt(variance.z()), mean.z() + sqrt(variance.z());
  }

  bool hasOverlap(SpatialDistributionCube<Scalar> sdc) {
    bool overlapX = (Cx[0] <= sdc.Cx[1] && Cx[1] >= sdc.Cx[0]);
    bool overlapY = (Cy[0] <= sdc.Cy[1] && Cy[1] >= sdc.Cy[0]);
    bool overlapZ = (Cz[0] <= sdc.Cz[1] && Cz[1] >= sdc.Cz[0]);

    return overlapX && overlapY && overlapZ;
  }

  Vec2 Cx, Cy, Cz;
};

struct MapUpdate {
  using Ptr = std::shared_ptr<MapUpdate>;
  Eigen::aligned_map<int64_t, Sophus::SE3f> keyframe_poses;
};

struct MapResponse {
  using Ptr = std::shared_ptr<MapResponse>;

  std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>> keyframe_poses;
  CovisibilityGraph::Ptr covisibility_graph;
  std::set<FrameId> not_marg_kfs;
};

struct MapIslandResponse {
  using Ptr = std::shared_ptr<MapIslandResponse>;

  std::vector<FrameId> island_keyframes;
  std::unordered_map<TimeCamId, Eigen::aligned_map<LandmarkId, Eigen::Matrix<double, 3, 1>>> landmarks_3d_map;
};

struct MapDatabaseVisualizationData {
  using Ptr = std::shared_ptr<MapDatabaseVisualizationData>;

  int64_t t_ns;

  Eigen::aligned_vector<Eigen::Vector3d> landmarks;
  std::vector<int> landmarks_ids;
  Eigen::aligned_map<FrameId, size_t> keyframe_idx;
  Eigen::aligned_map<int64_t, Sophus::SE3d> keyframe_poses;
  Eigen::aligned_vector<Eigen::Vector3d> covisibility;
  std::vector<int> covisibility_w;
  Eigen::aligned_vector<Eigen::Vector3d> spanning_tree;
  Eigen::aligned_vector<Eigen::Vector3d> loop_closures;
  std::map<int, Eigen::aligned_vector<Eigen::Vector3d>> observations;
};

// TODO: This should be a templated class template <class Scalar>
class MapDatabase {
 public:
  using Scalar = float;
  using Ptr = std::shared_ptr<MapDatabase>;

  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Vec4d = Eigen::Matrix<double, 4, 1>;
  using Vec4f = Eigen::Matrix<float, 4, 1>;
  using Mat4d = Eigen::Matrix<double, 4, 4>;

  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

  using SE3 = Sophus::SE3<Scalar>;

  MapDatabase(const VioConfig& config, const basalt::Calibration<double>& calib, const std::string& map_export_path);

  ~MapDatabase() { maybe_join(); }

  void initialize();

  void write_map_stamp(basalt::MapStamp::Ptr& map_stamp);

  void write_map_marg(std::set<FrameId>& keyframes_to_marg);

  void write_map_update(
      std::shared_ptr<Eigen::aligned_map<FrameId, Sophus::SE3f>>& keyframe_poses, FrameId candidate_kf_id,
      FrameId curr_kf_id, std::unordered_map<LandmarkId, LandmarkId>& lm_fusions,
      std::unordered_map<TimeCamId, std::unordered_map<LandmarkId, Eigen::Matrix<float, 2, 1>>>& curr_lc_obs);

  void read_covisibility_req(std::shared_ptr<std::vector<KeypointId>>& keypoints_ptr);

  void read_3d_points_req(FrameId keyframe, size_t neighbors_num);

  void read_map_req(FrameId frame_id);

  void get_map_points(Eigen::aligned_vector<Vec3d>& points, std::vector<int>& ids);

  Eigen::aligned_map<LandmarkId, Vec3d> get_landmarks_3d_pos(std::set<LandmarkId> landmarks);

  void computeMapVisualData();

  void handleCovisibilityReq(const std::vector<size_t>& curr_kpts);

  void computeSpatialDistributions(const std::set<TimeCamId>& kfs);

  void computeSTSMap(const std::vector<size_t>& curr_kpts);

  const std::map<std::string, double> getStats();

  void updateCovisibilityGraph(const LandmarkDatabase<Scalar>::Ptr& lmdb);

  void saveColmap(const std::string& path);

  void saveJson(const std::string& file_path);

  inline void maybe_join() {
    if (reading_thread) {
      reading_thread->join();
      reading_thread.reset();
    }
    if (writing_thread) {
      writing_thread->join();
      writing_thread.reset();
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  basalt::Calibration<double> calib;

  tbb::concurrent_bounded_queue<std::shared_ptr<WriteMessage>> write_queue;
  tbb::concurrent_bounded_queue<std::shared_ptr<ReadMessage>> read_queue;

  tbb::concurrent_bounded_queue<MapUpdate::Ptr>* out_map_update_queue = nullptr;
  tbb::concurrent_bounded_queue<MapDatabaseVisualizationData::Ptr>* out_vis_queue = nullptr;
  tbb::concurrent_bounded_queue<LandmarkDatabase<Scalar>::Ptr>* out_covi_res_queue = nullptr;
  tbb::concurrent_bounded_queue<MapResponse::Ptr>* out_map_res_queue;
  tbb::concurrent_bounded_queue<MapIslandResponse::Ptr>* out_3d_points_res_queue = nullptr;

  SyncState* sync_map_stamp = nullptr;
  SyncState* sync_lc_finished = nullptr;
  bool deterministic;

  std::shared_ptr<std::unordered_map<TimeCamId, std::string>> frame_id_to_name;

 private:
  std::string map_export_path;
  VioConfig config;
  std::shared_ptr<std::thread> reading_thread;
  std::shared_ptr<std::thread> writing_thread;
  MapDatabaseVisualizationData::Ptr map_visual_data;
  LandmarkDatabase<Scalar> map;
  std::mutex mutex;
  FrameId requested_frame_id = -1;
  CovisibilityGraph::Ptr covisibility_graph;

  std::set<FrameId> not_marg_kfs;

  // Covisibility
  Eigen::aligned_map<TimeCamId, SpatialDistributionCube<double>> keyframes_sdc;
  LandmarkDatabase<Scalar>::Ptr sts_map =
      std::make_shared<LandmarkDatabase<Scalar>>("STS Submap");  // spatial-temporal sensitive sub-global map
};
}  // namespace basalt
