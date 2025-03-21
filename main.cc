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

// #include <cstdlib>
// #include <iostream>
// #include <string>

// #include "absl/strings/str_join.h"
// #include "absl/time/time.h"
// #include "iopddl.h"
// #include "solver.h"

// int main(int argc, char* argv[]) {
//   const std::string filename = argv[1];
//   const absl::Duration timeout = absl::Seconds(atoi(argv[2]));
//   const auto problem = iopddl::ReadProblem(filename);
//   if (!problem.ok()) exit(1);
//   const auto solution = iopddl::Solver().Solve(*problem, timeout);
//   const auto result = solution.value_or(iopddl::Solution{});
//   std::cout << "[" << absl::StrJoin(result, ", ") << "]" << std::endl;

  
//   return 0;
// }
// main.cc
#include <cstdlib>
#include <iostream>
#include <string>
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "iopddl.h"
#include "solver.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_json> <timeout_seconds>\n";
    return 1;
  }
  std::string filename = argv[1];
  absl::Duration timeout = absl::Seconds(atoi(argv[2]));

  auto problemStatus = iopddl::ReadProblem(filename);
  if (!problemStatus.ok()) {
    std::cerr << "Error reading problem: " << problemStatus.status().ToString() << "\n";
    return 1;
  }
  iopddl::Problem problem = *problemStatus;

  auto solutionStatus = iopddl::Solver().Solve(problem, timeout);
  if (!solutionStatus.ok()) {
    std::cerr << "Error solving problem: " << solutionStatus.status().ToString() << "\n";
    return 1;
  }
  iopddl::Solution solution = *solutionStatus;
  std::cout << "Solution: [" << absl::StrJoin(solution, ", ") << "]\n";

  // using StrategyIdx = int64_t;
  // std::vector<iopddl::StrategyIdx> solution{0, 4, 4, 0, 3, 4, 0, 4, 4, 4, 3, 6, 8, 0, 8, 
  // 0, 4, 4, 0, 3, 4, 4, 6, 1, 1, 0, 2, 2, 2, 7, 1, 4, 0, 0, 0, 0, 0, 
  // 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 1, 3, 1, 1, 1, 2, 2, 2, 2, 3, 0, 0, 
  // 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 5, 1, 3, 1, 0, 0, 7, 0, 0, 0, 
  // 0, 0, 0, 1, 1, 1, 7, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 
  // 7, 0, 0, 30, 0, 0, 0, 0, 4, 0, 2, 6, 0, 0, 0, 0, 0, 0, 0, 6, 0, 
  // 2, 2, 6, 0, 5, 11, 0, 0, 21, 3, 3, 3, 0, 5, 3, 3, 0, 5, 3, 3, 13, 
  // 3, 13, 3, 3, 11, 3, 3, 4, 18, 0, 0, 0, 0, 0, 0, 6, 0, 2, 2, 6, 0, 
  // 0, 0, 0, 0, 0, 6, 0, 2, 2, 6, 0, 4, 0, 0, 0, 0, 0, 0, 7, 1, 1, 3, 
  // 1, 1, 1, 1, 0, 0, 0, 0, 4, 0, 0, 0, 7, 1, 1, 7, 1, 1, 1, 1, 0, 0, 
  // 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 6, 0, 0, 
  // 0, 0, 0, 0, 0, 6, 0, 2, 2, 6, 0, 5, 0, 0, 0, 11, 3, 3, 3, 3, 3, 3, 
  // 3, 3, 3, 3, 21, 3, 3, 4, 15, 0, 0, 0, 0, 0, 0, 6, 0, 2, 2, 6, 0, 0, 
  // 9, 2, 7, 1, 5, 5, 1, 1, 1, 1, 0, 0, 6, 0, 0, 6, 6, 0, 0, 0, 6, 6, 1, 
  // 1, 3, 2, 2, 2, 4, 5, 2, 4, 4, 2, 2, 2, 4, 5, 2, 4, 4, 0, 0, 0, 6, 6, 
  // 0, 0, 0, 6, 6, 1, 1, 3, 2, 2, 2, 4, 5, 2, 4, 4, 2, 2, 2, 4, 5, 2, 4, 
  // 4, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 6, 6, 1, 1, 3, 2, 2, 2, 4, 5, 2, 
  // 4, 4, 2, 2, 2, 4, 5, 2, 4, 4, 1, 0, 1, 1, 0, 1, 1, 1, 1, 3, 2, 2, 0, 
  // 2, 0, 1, 1, 0, 1, 1, 1, 5, 1, 1, 0, 2, 2, 2, 0, 1, 4, 20, 0, 0, 0, 6, 
  // 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  // 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 6, 2, 
  // 2, 2, 2, 0, 6, 6, 6, 6, 0, 6, 1, 1, 1, 1, 1, 1, 1, 3, 1, 0, 4, 0, 0, 
  // 4, 0, 6, 6, 6, 6, 5, 6, 0, 0, 6, 6, 6, 7, 3, 0, 6, 6, 0, 10, 3, 3, 3, 
  // 6, 3, 3, 0, 6, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 7, 0, 0, 
  // 0, 0, 0, 0, 5, 4, 5, 2, 0, 0, 0, 0, 0, 1, 1, 1, 4, 0, 2, 2, 0, 0, 5, 
  // 0, 0, 0, 5, 2, 2, 2, 0, 3, 2, 2, 0, 6, 2, 2, 6, 2, 6, 2, 2, 5, 2, 2, 
  // 4, 5, 0, 0, 0, 1, 1, 1, 4, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1, 4, 0, 2, 2,
  //  0, 0, 4, 6, 0, 0, 0, 0, 2, 1, 1, 5, 1, 1, 1, 1, 0, 0, 0, 0, 6, 0, 0,
  //  2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 8, 8, 
  //  6, 0, 0, 6, 6, 6, 6, 2, 2, 7, 5, 3, 3, 3, 3, 6, 3, 3, 3, 6, 0, 0, 0, 
  //  0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 7, 0, 0, 0, 0, 0, 5, 3, 4, 0, 2, 
  //  7, 0, 0, 0, 0, 1, 1, 1, 4, 0, 2, 2, 7, 0, 5, 0, 0, 0, 5, 2, 2, 2, 2, 
  //  2, 2, 2, 2, 2, 2, 5, 2, 2, 4, 6, 0, 0, 0, 1, 1, 1, 4, 0, 2, 2, 7, 0, 
  //  4, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 
  //  0, 0, 6, 6, 0, 0, 0, 6, 6, 1, 1, 3, 2, 2, 2, 4, 5, 2, 4, 4, 2, 2, 2, 
  //  4, 5, 2, 4, 4, 0, 0, 6, 6, 0, 0, 0, 6, 6, 1, 1, 3, 2, 2, 2, 4, 5, 2, 
  //  4, 4, 2, 2, 2, 4, 5, 2, 4, 4, 1};

  
  // Evaluate the solution:
  auto costStatus = iopddl::Evaluate(problem, solution);
  if (!costStatus.ok()) {
    std::cerr << "Error evaluating solution: " << costStatus.status().ToString() << "\n";
    return 1;
  }
  std::cout << "Total cost: " << *costStatus << "\n";

  return 0;
}
