// Copyright 2026, Mattis Krauch
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basalt/utils/build_config.h>

#ifdef BASALT_ENABLE_ROSBAG2
#include <mutex>
#include <regex>

#include <basalt/io/dataset_io.h>

#include <fastcdr/Cdr.h>
#include <sqlite3.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/highgui/highgui.hpp>

namespace basalt {
struct SqliteDeleter {
  void operator()(sqlite3* db) const {
    if (db) sqlite3_close(db);
  }
};

struct SqliteStmtDeleter {
  void operator()(sqlite3_stmt* stmt) const {
    if (stmt) sqlite3_finalize(stmt);
  }
};

using SqlitePtr = std::unique_ptr<sqlite3, SqliteDeleter>;
using SqliteStmtPtr = std::unique_ptr<sqlite3_stmt, SqliteStmtDeleter>;

class Rosbag2VioDataset : public VioDataset {
  size_t num_cams;
  std::string path;

  SqlitePtr db;
  SqliteStmtPtr stmt;
  std::mutex mtx;

  std::unordered_map<int64_t, std::vector<int64_t>> image_data_idx;

  std::vector<int64_t> image_timestamps;

  Eigen::aligned_vector<AccelData> accel_data;
  Eigen::aligned_vector<GyroData> gyro_data;

  std::vector<int64_t> gt_timestamps;
  Eigen::aligned_vector<Sophus::SE3d> gt_pose_data;

  int64_t mocap_to_imu_offset_ns = 0;
  std::vector<std::unordered_map<int64_t, double>> exposure_times;

 public:
  ~Rosbag2VioDataset() {}

  size_t get_num_cams() const { return num_cams; }
  std::vector<int64_t>& get_image_timestamps() { return image_timestamps; }

  const Eigen::aligned_vector<AccelData>& get_accel_data() const { return accel_data; }
  const Eigen::aligned_vector<GyroData>& get_gyro_data() const { return gyro_data; }
  const std::vector<int64_t>& get_gt_timestamps() const { return gt_timestamps; }
  const Eigen::aligned_vector<Sophus::SE3d>& get_gt_pose_data() const { return gt_pose_data; }

  int64_t get_mocap_to_imu_offset_ns() const override { return mocap_to_imu_offset_ns; }

  std::vector<ImageData> get_image_data(int64_t t_ns) {
    std::lock_guard<std::mutex> lock(mtx);

    std::vector<ImageData> res(num_cams);
    auto it = image_data_idx.find(t_ns);
    if (it == image_data_idx.end()) return res;
    const std::vector<int64_t>& rowids = it->second;

    for (size_t i = 0; i < num_cams; i++) {
      int64_t rowid = rowids[i];
      if (rowid == -1) continue;

      sqlite3_reset(stmt.get());
      sqlite3_bind_int64(stmt.get(), 1, rowid);

      if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        const void* blob_ptr = sqlite3_column_blob(stmt.get(), 0);
        int blob_size = sqlite3_column_bytes(stmt.get(), 0);

        eprosima::fastcdr::FastBuffer buffer((char*)blob_ptr, blob_size);
        eprosima::fastcdr::Cdr scdr(buffer);

        int32_t sec;
        uint32_t nanosec;
        std::string frame_id, format;

        // Read/Skip header
        scdr.read_encapsulation();
        scdr >> sec;
        scdr >> nanosec;
        scdr >> frame_id;
        scdr >> format;

        // Read blob to img
        std::vector<uint8_t> data;
        scdr >> data;
        cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);

        if (img.empty()) {
          std::cerr << "Error: cv::imdecode failed. jpeg_data size: " << data.size() << std::endl;
          continue;
        }

        if (img.type() == CV_8UC1) {
          res[i].img = std::make_shared<ManagedImage<uint16_t>>(img.cols, img.rows);

          const uint8_t* data_in = img.ptr();
          uint16_t* data_out = res[i].img->ptr;

          size_t full_size = long(img.cols) * img.rows;
          for (size_t j = 0; j < full_size; j++) {
            unsigned val = data_in[j];
            val = val << 8;
            data_out[j] = val;
          }
        } else if (img.type() == CV_8UC3) {
          res[i].img = std::make_shared<ManagedImage<uint16_t>>(img.cols, img.rows);

          const uint8_t* data_in = img.ptr();
          uint16_t* data_out = res[i].img->ptr;

          size_t full_size = long(img.cols) * img.rows;
          for (size_t j = 0; j < full_size; j++) {
            unsigned val = data_in[j * 3];
            val = val << 8;
            data_out[j] = val;
          }
        } else if (img.type() == CV_16UC1) {
          res[i].img = std::make_shared<ManagedImage<uint16_t>>(img.cols, img.rows);
          std::memcpy(res[i].img->ptr, img.ptr(), long(img.cols) * img.rows * sizeof(uint16_t));

        } else {
          std::cerr << "img.fmt.bpp " << img.type() << std::endl;
          std::abort();
        }
      }
      sqlite3_reset(stmt.get());
      sqlite3_clear_bindings(stmt.get());
    }
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class Rosbag2IO;
};

class Rosbag2IO : public DatasetIoInterface {
  std::map<std::string, int> cam_topics_to_id;
  std::string imu_topic;

  int n_timestamps, n_imu;

 public:
  Rosbag2IO() {}

  void read(const std::string& path) {
    if (!fs::exists(path)) std::cerr << "No dataset found in " << path << std::endl;

    data = std::make_shared<Rosbag2VioDataset>();
    sqlite3* handle = nullptr;
    sqlite3_open((path + "/rosbag/rosbag.db3").c_str(), &handle);
    data->db.reset(handle);

    const char* fetch_sql = "SELECT data FROM messages WHERE ROWID = ?;";
    sqlite3_stmt* stmt_ptr;
    sqlite3_prepare_v2(data->db.get(), fetch_sql, -1, &stmt_ptr, nullptr);
    data->stmt.reset(stmt_ptr);

    read_meta_data(path);
    data->num_cams = cam_topics_to_id.size();

    read_image_timestamps(path);
    read_image_indices(path);
    read_imu_data(path);
    read_gt_data_state(path);
  }

  void read_meta_data(const std::string& path) {
    YAML::Node config = YAML::LoadFile(path + "/rosbag/metadata.yaml");
    YAML::Node topics = config["rosbag2_bagfile_information"]["topics_with_message_count"];

    std::regex cam_regex("cam([0-9]+)");
    std::smatch match;

    if (topics.IsSequence()) {
      for (const auto& entry : topics) {
        if (entry["topic_metadata"] && entry["topic_metadata"]["name"]) {
          if (entry["topic_metadata"]["type"].as<std::string>() == "sensor_msgs/msg/CompressedImage") {
            std::string name = entry["topic_metadata"]["name"].as<std::string>();
            if (std::regex_search(name, match, cam_regex)) {
              int cam_id = std::stoi(match[1].str());
              cam_topics_to_id[name] = cam_id;
              n_timestamps = entry["message_count"].as<int>();
            }
          } else if (entry["topic_metadata"]["type"].as<std::string>() == "sensor_msgs/msg/Imu") {
            imu_topic = entry["topic_metadata"]["name"].as<std::string>();
            n_imu = entry["message_count"].as<int>();
          }
        }
      }
    }

    std::cout << "imu_topic: " << imu_topic << std::endl;
    std::cout << "cam_topics: ";
    for (const auto& [s, id] : cam_topics_to_id) std::cout << s << " ";
    std::cout << std::endl;
  }

  void read_image_timestamps(const std::string& /*path*/) {
    std::string sql =
        "SELECT m.timestamp FROM messages m "
        "JOIN topics t ON m.topic_id = t.id "
        "WHERE t.name = '" +
        cam_topics_to_id.begin()->first +
        "' "
        "ORDER BY m.timestamp ASC;";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(data->db.get(), sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
      data->image_timestamps.clear();
      data->image_timestamps.reserve(n_timestamps);

      while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t ts = sqlite3_column_int64(stmt, 0);
        data->image_timestamps.push_back(ts);
      }

      sqlite3_finalize(stmt);
    }
  }

  void read_image_indices(const std::string& /*path*/) {
    const char* sql =
        "SELECT m.rowid, m.timestamp, t.name FROM messages m "
        "JOIN topics t ON m.topic_id = t.id "
        "WHERE t.type = 'sensor_msgs/msg/CompressedImage' "
        "ORDER BY m.timestamp ASC, t.name ASC;";

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(data->db.get(), sql, -1, &stmt, nullptr);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
      int64_t rowid = sqlite3_column_int64(stmt, 0);
      int64_t ts = sqlite3_column_int64(stmt, 1);
      std::string topic_name = (const char*)sqlite3_column_text(stmt, 2);

      auto it = cam_topics_to_id.find(topic_name);
      if (it != cam_topics_to_id.end()) {
        int cam_idx = it->second;

        auto& row_vec = data->image_data_idx[ts];
        if (row_vec.empty()) row_vec.resize(data->num_cams, -1);
        row_vec[cam_idx] = rowid;
      }
    }

    sqlite3_finalize(stmt);
  }

  void read_imu_data(const std::string& /*path*/) {
    std::string sql =
        "SELECT m.data, m.timestamp FROM messages m "
        "JOIN topics t ON m.topic_id = t.id "
        "WHERE t.name = '" +
        imu_topic +
        "' "
        "ORDER BY m.timestamp ASC;";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(data->db.get(), sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
      data->accel_data.reserve(n_imu);
      data->gyro_data.reserve(n_imu);

      while (sqlite3_step(stmt) == SQLITE_ROW) {
        const void* blob_ptr = sqlite3_column_blob(stmt, 0);
        int blob_size = sqlite3_column_bytes(stmt, 0);
        int64_t db_timestamp = sqlite3_column_int64(stmt, 1);

        eprosima::fastcdr::FastBuffer fastbuffer((char*)blob_ptr, blob_size);
        eprosima::fastcdr::Cdr cdr_des(fastbuffer);

        cdr_des.read_encapsulation();

        // Header
        uint32_t sec, nanosec;
        std::string frame_id;
        cdr_des >> sec >> nanosec >> frame_id;

        // Orientation and covariance
        double ox, oy, oz, ow;
        cdr_des >> ox >> oy >> oz >> ow;
        for (int i = 0; i < 9; ++i) {
          double d;
          cdr_des >> d;
        }

        // Angular velocity and covariance
        double gx, gy, gz;
        cdr_des >> gx >> gy >> gz;
        for (int i = 0; i < 9; ++i) {
          double d;
          cdr_des >> d;
        }

        // Acceleration and covariance
        double ax, ay, az;
        cdr_des >> ax >> ay >> az;
        for (int i = 0; i < 9; ++i) {
          double d;
          cdr_des >> d;
        }

        data->accel_data.emplace_back();
        data->accel_data.back().timestamp_ns = db_timestamp;
        data->accel_data.back().data = Eigen::Vector3d(ax, ay, az);

        data->gyro_data.emplace_back();
        data->gyro_data.back().timestamp_ns = db_timestamp;
        data->gyro_data.back().data = Eigen::Vector3d(gx, gy, gz);
      }
      sqlite3_finalize(stmt);
    }
  }

  void read_gt_data_state(const std::string& path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    std::ifstream f(path + "/groundtruth.txt");
    std::string line;

    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;

      std::stringstream ss(line);
      double tx, ty, tz, qx, qy, qz, qw;
      std::string ts_str;

      if (!(ss >> ts_str >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;

      ts_str.erase(std::remove(ts_str.begin(), ts_str.end(), '.'), ts_str.end());
      uint64_t timestamp_int = std::stoull(ts_str);

      Eigen::Quaterniond q(qw, qx, qy, qz);
      Eigen::Vector3d pos(tx, ty, tz);

      data->gt_timestamps.emplace_back(timestamp_int);
      data->gt_pose_data.emplace_back(q, pos);
    }
  }

  void reset() { data.reset(); }

  VioDatasetPtr get_data() { return data; }

 private:
  std::shared_ptr<Rosbag2VioDataset> data;
};

}  // namespace basalt

#endif  // BASALT_ENABLE_ROSBAG2
