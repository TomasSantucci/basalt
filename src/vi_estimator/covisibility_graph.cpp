#include <basalt/vi_estimator/covisibility_graph.h>

namespace basalt {

const std::unordered_map<FrameId, int>& CovisibilityGraph::getCovisibility(FrameId id) const { return covis_.at(id); }

const FrameId& CovisibilityGraph::getParentId(FrameId id) const { return tree_.at(id).parent; }

const std::vector<FrameId>& CovisibilityGraph::getChildrenIds(FrameId id) const { return tree_.at(id).children; }

const std::unordered_map<FrameId, std::vector<FrameId>>& CovisibilityGraph::getLoopClosures() const {
  return loop_closures_;
}

bool CovisibilityGraph::hasNode(FrameId id) const { return covis_.find(id) != covis_.end(); }

bool CovisibilityGraph::edgeExists(FrameId id1, FrameId id2) const {
  auto it = covis_.find(id1);
  if (it == covis_.end()) return false;
  return it->second.find(id2) != it->second.end();
}

void CovisibilityGraph::setEdge(FrameId id1, FrameId id2, int weight) {
  covis_[id1][id2] = weight;
  covis_[id2][id1] = weight;
}

void CovisibilityGraph::incrementEdge(FrameId id1, FrameId id2, int weight) {
  covis_[id1][id2] += weight;
  covis_[id2][id1] += weight;
}

void CovisibilityGraph::addTreeNode(FrameId id, FrameId parent_id) {
  TreeNode node;
  node.parent = parent_id;
  tree_[id] = node;

  if (parent_id != invalid()) {
    tree_[parent_id].children.push_back(id);
  }
}

void CovisibilityGraph::addLoopClosure(FrameId id1, FrameId id2) {
  loop_closures_[id1].push_back(id2);
  loop_closures_[id2].push_back(id1);
}

void CovisibilityGraph::setRoot(FrameId id) { root_ = id; }

FrameId CovisibilityGraph::getRoot() const { return root_; }

void CovisibilityGraph::removeNode(FrameId id) {
  // Removing the root is not allowed for now
  BASALT_ASSERT(id != getRoot());

  auto covi_it = covis_.find(id);
  if (covi_it == covis_.end()) return;

  for (const auto& [other_id, neighbors] : covi_it->second) {
    covis_[other_id].erase(id);
  }
  covis_.erase(covi_it);

  // Remove node from the spanning tree
  auto tree_it = tree_.find(id);
  if (tree_it != tree_.end()) {
    const FrameId parent_id = tree_it->second.parent;

    // Remove node from the parent's children list
    if (parent_id != invalid()) {
      auto& siblings = tree_[parent_id].children;
      siblings.erase(std::remove(siblings.begin(), siblings.end(), id), siblings.end());
    }

    // Reassign children to the parent
    for (const auto& child_id : tree_it->second.children) {
      tree_[child_id].parent = parent_id;
      if (parent_id != invalid()) {
        tree_[parent_id].children.push_back(child_id);
      }
    }
    tree_.erase(tree_it);
  }

  // Remove loop closures
  auto lc_it = loop_closures_.find(id);
  if (lc_it != loop_closures_.end()) {
    for (const auto& other_id : lc_it->second) {
      auto& closures = loop_closures_[other_id];
      closures.erase(std::remove(closures.begin(), closures.end(), id), closures.end());
    }
    loop_closures_.erase(lc_it);
  }
}

std::vector<FrameId> CovisibilityGraph::getTopCovisible(FrameId id, size_t k) const {
  auto it = covis_.find(id);
  if (it == covis_.end()) {
    return {};
  }

  std::vector<std::pair<FrameId, int>> neighbors;
  for (const auto& [neighbor_id, weight] : it->second) {
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

std::vector<FrameId> CovisibilityGraph::getCovisibleAbove(FrameId id, int weight_threshold) const {
  auto it = covis_.find(id);
  if (it == covis_.end()) {
    return {};
  }

  std::vector<FrameId> result;
  for (const auto& [neighbor_id, weight] : it->second) {
    if (weight >= weight_threshold) {
      result.push_back(neighbor_id);
    }
  }
  return result;
}

void CovisibilityGraph::print_stats() const {
  std::cout << "Covisibility Graph Stats:" << std::endl;
  std::cout << "  Number of nodes: " << covis_.size() << std::endl;

  size_t total_edges = 0;
  for (const auto& [id, neighbors] : covis_) {
    total_edges += neighbors.size();
  }
  // Each edge is counted twice
  total_edges /= 2;
  std::cout << "  Number of edges: " << total_edges << std::endl;

  std::cout << "  Number of tree nodes: " << tree_.size() << std::endl;

  size_t total_loop_closures = 0;
  for (const auto& [id, closures] : loop_closures_) {
    total_loop_closures += closures.size();
  }
  // Each loop closure is counted twice
  total_loop_closures /= 2;
  std::cout << "  Number of loop closures: " << total_loop_closures << std::endl;
}

}  // namespace basalt
