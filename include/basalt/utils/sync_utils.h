#pragma once

#include <condition_variable>

namespace basalt {

struct SyncState {
  std::mutex m;
  std::condition_variable cvar;
  bool ready = false;
};

}  // namespace basalt
