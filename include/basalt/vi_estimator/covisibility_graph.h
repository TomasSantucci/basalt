#pragma once

#include <basalt/utils/common_types.h>

namespace basalt {

class CovisibilityGraph {
 public:
  using Ptr = std::shared_ptr<CovisibilityGraph>;
  static constexpr FrameId invalid() { return FrameId(-1); }

  const std::unordered_map<FrameId, int>& getCovisibility(FrameId id) const;

  const FrameId& getParentId(FrameId id) const;

  const std::vector<FrameId>& getChildrenIds(FrameId id) const;

  const std::unordered_map<FrameId, std::vector<FrameId>>& getLoopClosures() const;

  bool hasNode(FrameId id) const;

  bool edgeExists(FrameId id1, FrameId id2) const;

  void setEdge(FrameId id1, FrameId id2, int weight);

  void incrementEdge(FrameId id1, FrameId id2, int weight);

  void addTreeNode(FrameId id, FrameId parent_id);

  void addLoopClosure(FrameId id1, FrameId id2);

  void setRoot(FrameId id);

  FrameId getRoot() const;

  void removeNode(FrameId id);

  std::vector<FrameId> getTopCovisible(FrameId id, size_t k) const;

  std::vector<FrameId> getCovisibleAbove(FrameId id, int weight_threshold) const;

  void print_stats() const;

 private:
  struct TreeNode {
    FrameId parent = invalid();
    std::vector<FrameId> children;
  };

  std::unordered_map<FrameId, std::unordered_map<FrameId, int>> covis_;

  std::unordered_map<FrameId, TreeNode> tree_;

  std::unordered_map<FrameId, std::vector<FrameId>> loop_closures_;

  FrameId root_ = invalid();
};

}  // namespace basalt
