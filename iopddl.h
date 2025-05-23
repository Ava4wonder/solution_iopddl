/*
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef IOPDDL_IOPDDL_H_
#define IOPDDL_IOPDDL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <iostream>
#include <unordered_map>
#include <map>
#include <queue>
#include <limits>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"

////////////////////////////////////////////////////////////////////////////////
/////////  Basic definitions for problem & solution data structures.   /////////
/////////  Contest participants do not need to modify this code.       /////////
////////////////////////////////////////////////////////////////////////////////

namespace iopddl {

using Cost = double;
// using Usage = int64_t
using Usage = double;
using TimeIdx = int64_t;
using NodeIdx = int64_t;
using EdgeIdx = int64_t;
using StrategyIdx = int64_t;
using Interval = std::pair<TimeIdx, TimeIdx>;
using Solution = std::vector<StrategyIdx>;
// using TotalUsage = absl::int128;
using TotalUsage = double;
using TotalCost = double;

struct Strategy {
  Cost cost;
  Usage usage;
  bool operator==(const Strategy& other) const = default;
};

struct Node {
  Interval interval;  // Interpreted as half-open with an exclusive upper bound
  std::vector<Strategy> strategies;
  bool operator==(const Node& other) const = default;
};

struct Edge {
  using Nodes = std::vector<NodeIdx>;
  Nodes nodes;
  std::vector<Strategy> strategies;
  bool operator==(const Edge& other) const = default;
};

struct Problem {
  std::string name;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::optional<Usage> usage_limit;
  bool operator==(const Problem& other) const = default;
};


absl::StatusOr<TotalCost> Evaluate(const Problem& problem,
                                   const Solution& solution);

absl::StatusOr<Problem> ReadProblem(const std::string& filename);

}  // namespace iopddl

#endif  // IOPDDL_IOPDDL_H_
