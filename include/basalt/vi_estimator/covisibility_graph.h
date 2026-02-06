#pragma once

#include <basalt/utils/common_types.h>

namespace basalt {

class CovisibilityGraph {
 public:
  using Ptr = std::shared_ptr<CovisibilityGraph>;
  static constexpr FrameId invalid() { return FrameId(-1); }

  struct TreeNode {
    FrameId parent = invalid();
    std::vector<FrameId> children;
  };

  const std::unordered_map<FrameId, int>& getCovisibleKfs(FrameId id) const { return covis_.at(id); }

  const FrameId& getParentNode(FrameId id) const { return tree_.at(id).parent; }

  const std::vector<FrameId>& getChildrenNodes(FrameId id) const { return tree_.at(id).children; }

  const std::vector<FrameId>& getLoopClosures(FrameId id) const { return loop_closures_.at(id); }

  const std::unordered_map<FrameId, std::vector<FrameId>>& getAllLoopClosures() const { return loop_closures_; }

  // TODO@tsantucci: add a hasTreeNode method too
  const bool hasNode(FrameId id) const { return covis_.find(id) != covis_.end(); }

  const bool edgeExists(FrameId id1, FrameId id2) const {
    auto it = covis_.find(id1);
    if (it == covis_.end()) return false;
    return it->second.find(id2) != it->second.end();
  }

  void addEdge(FrameId id1, FrameId id2, int weight) {
    covis_[id1][id2] = weight;
    covis_[id2][id1] = weight;
  }

  void updateEdge(FrameId id1, FrameId id2, int weight) {
    covis_[id1][id2] += weight;
    covis_[id2][id1] += weight;
  }

  std::vector<FrameId> getTopK(FrameId id, size_t k) const {
    std::vector<std::pair<FrameId, int>> neighbors;
    for (const auto& [neighbor_id, weight] : covis_.at(id)) {
      neighbors.emplace_back(neighbor_id, weight);
    }

    std::sort(neighbors.begin(), neighbors.end(),
              [](const std::pair<FrameId, int>& a, const std::pair<FrameId, int>& b) { return a.second > b.second; });

    std::vector<FrameId> top_k;
    for (size_t i = 0; i < std::min(k, neighbors.size()); i++) {
      top_k.push_back(neighbors[i].first);
    }

    return top_k;
  }

  std::vector<FrameId> getAboveWeight(FrameId id, int weight_threshold) const {
    std::vector<FrameId> result;
    for (const auto& [neighbor_id, weight] : covis_.at(id)) {
      if (weight >= weight_threshold) {
        result.push_back(neighbor_id);
      }
    }
    return result;
  }

  void removeNode(FrameId id) {
    for (auto& [other_id, neighbors] : covis_[id]) {
      covis_[other_id].erase(id);
    }
    covis_.erase(id);

    // also remove from tree

    // remove in case there's a loop closure
  }

  // i dont even know if we need this
  void setRoot(FrameId id) { root_ = id; }

  FrameId getRoot() const { return root_; }

  void addTreeNode(FrameId id, FrameId parent_id) {
    TreeNode node;
    node.parent = parent_id;
    tree_[id] = node;

    if (parent_id != invalid()) {
      tree_[parent_id].children.push_back(id);
    }
  }

  void addLoopClosure(FrameId id1, FrameId id2) {
    loop_closures_[id1].push_back(id2);
    loop_closures_[id2].push_back(id1);
  }

  void print() const {
    std::cout << "Covisibility Graph:" << std::endl;
    for (const auto& [id, neighbors] : covis_) {
      std::cout << "  Node " << id << " connected to: ";
      for (const auto& [neighbor_id, weight] : neighbors) {
        std::cout << "(" << neighbor_id << ", weight: " << weight << ") ";
      }
      std::cout << std::endl;
    }

    std::cout << "Spanning Tree:" << std::endl;
    for (const auto& [id, node] : tree_) {
      std::cout << "  Node " << id << " with parent " << node.parent << " and children: ";
      for (const auto& child_id : node.children) {
        std::cout << child_id << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "Loop Closures:" << std::endl;
    for (const auto& [id, closures] : loop_closures_) {
      std::cout << "  Node " << id << " has loop closures with: ";
      for (const auto& closure_id : closures) {
        std::cout << closure_id << " ";
      }
      std::cout << std::endl;
    }
  }

  void print_stats() const {
    std::cout << "Covisibility Graph Stats:" << std::endl;
    std::cout << "  Number of nodes: " << covis_.size() << std::endl;

    size_t total_edges = 0;
    for (const auto& [id, neighbors] : covis_) {
      total_edges += neighbors.size();
    }
    // each edge is counted twice
    total_edges /= 2;
    std::cout << "  Number of edges: " << total_edges << std::endl;

    std::cout << "  Number of tree nodes: " << tree_.size() << std::endl;

    size_t total_loop_closures = 0;
    for (const auto& [id, closures] : loop_closures_) {
      total_loop_closures += closures.size();
    }
    // each loop closure is counted twice
    total_loop_closures /= 2;
    std::cout << "  Number of loop closures: " << total_loop_closures << std::endl;
  }

 private:
  // Covisibility graph
  std::unordered_map<FrameId, std::unordered_map<FrameId, int>> covis_;

  // Spanning tree
  std::unordered_map<FrameId, TreeNode> tree_;

  // Loop closures
  std::unordered_map<FrameId, std::vector<FrameId>> loop_closures_;

  FrameId root_ = invalid();
};

}  // namespace basalt
