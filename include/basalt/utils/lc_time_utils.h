#pragma once

#include <chrono>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>

namespace basalt {
enum class LCTimeStage {
  Start,
  HashBowIndex,
  HashBowSearch,
  InitialMatch,
  LandmarksRequest,
  IslandMatch,
  GeometricVerification,
  Reprojection,
  ReprojectedGeometricVerification,
  LoopClosure
};

struct LCTimeStats {
  int64_t current_kf_ts = -1;
  int64_t start_ts;

  int64_t hash_bow_index_time;

  int64_t hash_bow_search_time;

  std::vector<int64_t> initial_match_time;
  std::vector<int64_t> landmarks_request_time;
  std::vector<int64_t> island_match_time;
  std::vector<int64_t> geometric_verification_time;
  std::vector<int64_t> reprojection_time;
  std::vector<int64_t> reprojected_geometric_verification_time;

  int64_t loop_closure_time;

  bool loop_closed = false;

  void resetStats() {
    start_ts = -1;
    hash_bow_index_time = -1;
    hash_bow_search_time = -1;
    initial_match_time.clear();
    landmarks_request_time.clear();
    island_match_time.clear();
    geometric_verification_time.clear();
    reprojection_time.clear();
    reprojected_geometric_verification_time.clear();
    loop_closure_time = -1;
    loop_closed = false;
  }

  void addTime(LCTimeStage name, bool success) {
    int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();

    switch (name) {
      case LCTimeStage::Start:
        start_ts = now;
        break;
      case LCTimeStage::HashBowIndex:
        hash_bow_index_time = now;
        break;
      case LCTimeStage::HashBowSearch:
        hash_bow_search_time = now;
        break;
      case LCTimeStage::InitialMatch:
        initial_match_time.push_back(now);
        if (!success) landmarks_request_time.push_back(-1);
        break;
      case LCTimeStage::LandmarksRequest:
        landmarks_request_time.push_back(now);
        if (!success) island_match_time.push_back(-1);
        break;
      case LCTimeStage::IslandMatch:
        island_match_time.push_back(now);
        if (!success) geometric_verification_time.push_back(-1);
        break;
      case LCTimeStage::GeometricVerification:
        geometric_verification_time.push_back(now);
        if (!success) reprojection_time.push_back(-1);
        break;
      case LCTimeStage::Reprojection:
        reprojection_time.push_back(now);
        if (!success) reprojected_geometric_verification_time.push_back(-1);
        break;
      case LCTimeStage::ReprojectedGeometricVerification:
        reprojected_geometric_verification_time.push_back(now);
        break;
      case LCTimeStage::LoopClosure:
        loop_closure_time = now;
        break;
    }
  }

  std::ostream& dumpHeader(std::ostream& os) const {
    os << "current_kf_ts,start_ts,hash_bow_index_time,hash_bow_search_time,initial_match_time,"
          "landmarks_request_time,island_match_time,geometric_verification_time,reprojection_time,"
          "reprojected_geometric_verification_time,loop_closure_time,loop_closed\n";
    return os;
  }
};

inline std::ostream& operator<<(std::ostream& os, const LCTimeStats& stats) {
  os << stats.current_kf_ts << "," << stats.start_ts << "," << stats.hash_bow_index_time << ","
     << stats.hash_bow_search_time << ",";

  std::ostringstream initial_match_string;
  for (size_t i = 0; i < stats.initial_match_time.size(); i++) {
    initial_match_string << stats.initial_match_time[i];
    if (i + 1 < stats.initial_match_time.size()) {
      initial_match_string << ";";
    }
  }
  os << "\"" << initial_match_string.str() << "\"" << ",";

  std::ostringstream landmarks_request_string;
  for (size_t i = 0; i < stats.landmarks_request_time.size(); i++) {
    landmarks_request_string << stats.landmarks_request_time[i];
    if (i + 1 < stats.landmarks_request_time.size()) {
      landmarks_request_string << ";";
    }
  }
  os << "\"" << landmarks_request_string.str() << "\"" << ",";

  std::ostringstream island_match_string;
  for (size_t i = 0; i < stats.island_match_time.size(); i++) {
    island_match_string << stats.island_match_time[i];
    if (i + 1 < stats.island_match_time.size()) {
      island_match_string << ";";
    }
  }
  os << "\"" << island_match_string.str() << "\"" << ",";

  std::ostringstream geometric_verification_string;
  for (size_t i = 0; i < stats.geometric_verification_time.size(); i++) {
    geometric_verification_string << stats.geometric_verification_time[i];
    if (i + 1 < stats.geometric_verification_time.size()) {
      geometric_verification_string << ";";
    }
  }
  os << "\"" << geometric_verification_string.str() << "\"" << ",";

  std::ostringstream reprojection_string;
  for (size_t i = 0; i < stats.reprojection_time.size(); i++) {
    reprojection_string << stats.reprojection_time[i];
    if (i + 1 < stats.reprojection_time.size()) {
      reprojection_string << ";";
    }
  }
  os << "\"" << reprojection_string.str() << "\"" << ",";

  std::ostringstream reprojected_geometric_verification_string;
  for (size_t i = 0; i < stats.reprojected_geometric_verification_time.size(); i++) {
    reprojected_geometric_verification_string << stats.reprojected_geometric_verification_time[i];
    if (i + 1 < stats.reprojected_geometric_verification_time.size()) {
      reprojected_geometric_verification_string << ";";
    }
  }
  os << "\"" << reprojected_geometric_verification_string.str() << "\"" << ",";

  os << stats.loop_closure_time << "," << (stats.loop_closed ? 1 : 0) << "\n";
  return os;
}

}  // namespace basalt
