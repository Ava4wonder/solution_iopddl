// scso_solver.cc
#include "scso_solver.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "absl/strings/str_join.h"


namespace iopddl {

// Helper function: compute all unions
std::pair<std::vector<std::set<int>>, std::unordered_map<int, std::vector<int>>> ComputeAllUnions(const Problem &problem, const std::vector<int> &criticalNodes) {
  struct Event {
    int time;
    bool isStart;  // true for start, false for end
    int node;
  };

  std::vector<Event> events;
  for (int i = 0; i < problem.nodes.size(); ++i) {
    const auto &interval = problem.nodes[i].interval;
    if (interval.first == interval.second)
      continue;
    events.push_back({static_cast<int>(interval.first), true, i});
    events.push_back({static_cast<int>(interval.second), false, i});
    // events.push_back({interval.first, true, i});
    // events.push_back({interval.second, false, i});
  }
  std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
    if (a.time == b.time)
      return (!a.isStart && b.isStart);  // process end events before start events
    return a.time < b.time;
  });

  std::set<int> active;
  int prev_time = -1;
  std::vector<std::set<int>> activeSets;
  for (const auto &e : events) {
    if (prev_time != -1 && e.time != prev_time && !active.empty()) {
      activeSets.push_back(active);
    }
    if (e.isStart)
      active.insert(e.node);
    else
      active.erase(e.node);
    prev_time = e.time;
  }
  // Remove duplicate sets.
  std::vector<std::set<int>> uniqueSets;
  for (const auto &s : activeSets) {
    if (std::find(uniqueSets.begin(), uniqueSets.end(), s) == uniqueSets.end())
      uniqueSets.push_back(s);
  }

  // Create node-to-union mapping
    std::unordered_map<int, std::vector<int>> node_to_unions;
    for (int node : criticalNodes) {
        node_to_unions[node] = {};
    }

    for (size_t union_idx = 0; union_idx < uniqueSets.size(); ++union_idx) {
        for (int node : uniqueSets[union_idx]) {
            if (node_to_unions.find(node) != node_to_unions.end()) {
                node_to_unions[node].push_back(union_idx);
            }
        }
    }

    return {uniqueSets, node_to_unions};
}

std::unordered_map<int, int> CheckSafestSolutionFeasible(
    const Solution &solution,
    const std::vector<std::set<int>> &allUnions,
    const Problem &problem) {
  
  if (!problem.usage_limit)
    return {};  // No limit means always feasible.

  double limit = *problem.usage_limit;
  std::unordered_map<int, int> union_mem_dict;

  for (size_t uidx = 0; uidx < allUnions.size(); ++uidx) {
    double totalUsage = 0;
    for (int node : allUnions[uidx]) {
      totalUsage += problem.nodes[node].strategies[solution.at(node)].usage;
    }
    if (totalUsage > limit) {
      std::cout << " >>> infeasible union <<< " 
            << absl::StrJoin(allUnions[uidx], ", ") 
            << std::endl;
      std::cout << " >>> totalUsage: <<< " << totalUsage << "; limit: " << limit
            << std::endl;
      for (int node : allUnions[uidx]) {
        std::cout << "i: " << node << ";" << problem.nodes[node].strategies[solution.at(node)].usage << std::endl;
      }
      return {};  // Indicates infeasible solution
    }
    union_mem_dict[uidx] = totalUsage;
  }
  
  return union_mem_dict;
}

// Function to check if modifying a node maintains feasibility
bool IsFeasibleUpdated(
    int node_index,
    int pre_option_idx,
    Solution &solution,
    std::unordered_map<int, int> &union_mem_dict,
    const std::unordered_map<int, std::vector<int>> &node_to_unions,
    const Problem &problem) {

  if (!problem.usage_limit)
    return true;

  double limit = *problem.usage_limit;
  const auto &macro_usages = problem.nodes[node_index].strategies;

  if (node_to_unions.find(node_index) == node_to_unions.end()) {
    return true;  // Node does not belong to any union
  }

  for (int uindex : node_to_unions.at(node_index)) {
    double cur_mem_usage = union_mem_dict[uindex] 
                        - macro_usages[pre_option_idx].usage 
                        + macro_usages[solution.at(node_index)].usage;

    if (cur_mem_usage > limit) {
        if((node_index == 14339) && (solution[node_index] == 8)) {
            std::cout << "Infeasible at union: " << uindex << "pre_usgae" << union_mem_dict[uindex] << "now" << cur_mem_usage << std::endl;
        }
    //   std::cout << "#### Node Index: " << node_index << std::endl;
      return false;
    }
    union_mem_dict[uindex] = cur_mem_usage;
  }

  return true;
}

// Helper: check if a candidate solution is feasible.
// For each parent union, the sum of the usages (of the chosen strategies) must not exceed the usage limit.
bool IsFeasible(const Solution &solution,
                const std::vector<std::set<int>> &parentUnions,
                const Problem &problem) {
  if (!problem.usage_limit)
    return true;  // No limit means always feasible.
  int limit = *problem.usage_limit;
  for (const auto &unionSet : parentUnions) {
    int totalUsage = 0;
    for (int node : unionSet) {
      totalUsage += problem.nodes[node].strategies[solution[node]].usage;
    }
    if (totalUsage > limit)
      return false;
  }
  return true;
}

// Helper: compute the total cost of a candidate solution.
// (Note: This re-implements the cost computation without the time-step usage check.)
double ComputeCost(const Solution &solution, const Problem &problem) {
  double totalCost = 0;
  for (int i = 0; i < solution.size(); ++i) {
    totalCost += problem.nodes[i].strategies[solution[i]].cost;
  }
  for (const auto &edge : problem.edges) {
    int strategy_idx = 0;
    for (const int node : edge.nodes) {
      strategy_idx = strategy_idx * problem.nodes[node].strategies.size() + solution[node];
    }
    totalCost += edge.strategies[strategy_idx].cost;
  }
  return totalCost;
}

// Define the function to compute criticality and minCostOptIndex
// std::pair<std::vector<double>, std::unordered_map<int, int>> ComputeCriticality(
std::tuple<std::vector<double>, std::vector<double>, std::unordered_map<int, int>> ComputeCriticality(
    const Problem &ComputeCriticality,
    const Solution &safestSolution,
    double safestCost) {

  std::vector<double> crit(problem.nodes.size(), 0.0);
  std::vector<double> sigcrit(problem.nodes.size(), 0.0);
  std::unordered_map<int, int> minCostOptIndex;  // Stores the min cost option index for each node

  // Variable to store the cost of the current safest option
  double CurOptCost = 0.0;

  // For each node (macro-node) in the problem.
  for (size_t i = 0; i < problem.nodes.size(); ++i) {
    const auto &node = problem.nodes[i];
    std::vector<double> optionCosts;  // Option cost for each strategy of node i.
    
    // For each available option (strategy) for node i.
    for (size_t opt = 0; opt < node.strategies.size(); ++opt) {
      // Start with the base cost of this option.
      double cost = node.strategies[opt].cost;
      
      // Process each edge in the problem.
      for (const auto &edge : problem.edges) {
        // Check if node i is the source in this edge.
        if (!edge.nodes.empty() && edge.nodes[0] == i && edge.nodes.size() >= 2) {
          int dest = edge.nodes[1];
          if (dest < problem.nodes.size()) {
            int q = safestSolution[dest];
            size_t numStrategiesDest = problem.nodes[dest].strategies.size();
            int index = static_cast<int>(opt * numStrategiesDest + q);
            if (index < edge.strategies.size()) {
              cost += edge.strategies[index].cost;
            }
          }
        }
        // Check if node i is the destination in this edge.
        if (!edge.nodes.empty() && edge.nodes.size() >= 2 && edge.nodes[1] == i) {
          int src = edge.nodes[0];
          if (src < problem.nodes.size()) {
            int p = safestSolution[src];
            size_t numStrategiesCurrent = node.strategies.size();
            int index = static_cast<int>(p * numStrategiesCurrent + opt);
            if (index < edge.strategies.size()) {
              cost += edge.strategies[index].cost;
            }
          }
        }
      }
      
      optionCosts.push_back(cost);
      if (i==15738 || i == 15743){
        std::cout << "node: " << i << "option: " << opt << "cost: " << cost << "cur_opt" << safestSolution[i] << std::endl; 
        }

      // Store the cost of the current safest solution option
      if (opt == static_cast<size_t>(safestSolution[i])) {
        CurOptCost = cost;
      }
    }
    
    // Compute the minimum and maximum option costs for node i.
    double minCost = *std::min_element(optionCosts.begin(), optionCosts.end());
    // double maxCost = *std::max_element(optionCosts.begin(), optionCosts.end());
    // double costRange = maxCost - minCost;
    double costRange = CurOptCost - minCost;

    
    // Determine the index of the minimal cost option.
    size_t minIndex = std::distance(optionCosts.begin(),
                                    std::find(optionCosts.begin(), optionCosts.end(), minCost));

    // Store the min cost option index
    minCostOptIndex[i] = static_cast<int>(minIndex);

    // If the safest solution for node i picks the option with the minimal cost, assign criticality 0.
    if (static_cast<size_t>(safestSolution[i]) == minIndex) {
      crit[i] = 0.0;
    } else {
      if (CurOptCost == 0 || safestCost == 0)
        crit[i] = 0.0;
      else
        crit[i] = (static_cast<double>(costRange) / CurOptCost) *
                  (static_cast<double>(CurOptCost) / safestCost);
    }
    if (i==15738 || i == 15743){
        std::cout << "node: " << i << "crit score: " << crit[i] << std::endl; 
        }
    if (minCost >= 1e18){
        sigcrit[i] = 1.0;
        // crit[i] += 0.0001;
        std::cout << "node: " << i << "mincost " << minCost << "crit score:" << crit[i] <<std::endl; 
        }
  }
  
  return {crit, sigcrit, minCostOptIndex};  // Return both crit values and minCost option indices
}



// Main SCSO routine: runs the optimization and returns the best solution found.
// Solution RunSCSO(const Problem &problem) {
Solution RunSCSO(const Problem &problem, absl::Time start_time, absl::Duration timeout) {
  // Compute the “safest solution”: for each node, choose the option with minimal usage.
  Solution safestSolution;
  for (const auto &node : problem.nodes) {
    int bestOpt = 0;
    double minUsage = node.strategies[0].usage;
    for (size_t j = 1; j < node.strategies.size(); ++j) {
      if (node.strategies[j].usage < minUsage) {
        minUsage = node.strategies[j].usage;
        bestOpt = j;
      }
    }
    safestSolution.push_back(bestOpt);
  }
  double safestCost = ComputeCost(safestSolution, problem);
  std::cout << "Safest solution cost: " << safestCost << std::endl;

  Solution bestSolution = safestSolution;
  double bestCost = safestCost;

  // std::cout << " >>> node 19469::14339  <<< " 
  //           << bestSolution[19469] << bestSolution[14339]
  //           << std::endl;

  // Compute all unions from node intervals.
  std::vector<int> allNodes(problem.nodes.size());
  std::iota(allNodes.begin(), allNodes.end(), 0);
  std::cout << " >>> Start ComputeParentUnions <<< " << std::endl;
  auto [allUnions, nodeToUnions] = ComputeAllUnions(problem, allNodes);
  std::cout << " >>> ComputeParentUnions DONE <<< " << std::endl;
  std::unordered_map<int, int> union_mem_dict = CheckSafestSolutionFeasible(safestSolution, allUnions, problem);

  if (union_mem_dict.empty()) {
      std::cout << "The safest solution violates the memory usage limit in some parent union!" << std::endl;
      throw std::runtime_error("Safest solution is not feasible");
  }
  
  
  // Enter criticality-based search-and-optimize loop
  int max_criticality_recompute_times = 2;
  std::vector<int> unchangedCriticalNodes;

  for (int restart_count = 0; restart_count < max_criticality_recompute_times; ++restart_count) {
    std::cout << " @@ criticality_recompute @@ <<< " << restart_count << std::endl;
    // Compute criticality scores.
    auto [critScores, sigcrit, minCostIndices] = ComputeCriticality(problem, bestSolution, bestCost);  
    // Sort nodes in descending order of criticality.
    std::vector<int> sortedNodes = allNodes;
    std::sort(sortedNodes.begin(), sortedNodes.end(), [&](int a, int b) {
        return critScores[a] > critScores[b];
    });

    int numCritical = std::max(1, static_cast<int>(0.4 * problem.nodes.size()));
    std::vector<int> criticalNodes(sortedNodes.begin(), sortedNodes.begin() + numCritical);

    // Create a copy to track unchanged nodes
    unchangedCriticalNodes = criticalNodes;
    //   std::vector<int> criticalNodes{541, 694, 579, 590, 726, 488, 642, 649, 276, 106, 736, 196, 247, 388, 132, 555, 187, 495, 535, 389, 262, 413, 576, 45, 203, 738, 601, 572, 48, 409, 258, 469, 708, 149, 87, 424, 421, 621, 606, 676, 200, 12, 723, 719, 470, 184, 609, 624, 500, 473, 84, 523, 643, 499, 650, 14, 740, 153, 463, 663, 526, 510, 652, 509, 679, 502, 659, 653, 71, 74, 229, 146, 496, 144, 114, 164, 506, 477, 273, 483, 598, 569, 567, 452, 241, 11, 141, 446, 480, 545, 16, 686};
    std::cout << " >>> critical nodes <<< " 
            << absl::StrJoin(criticalNodes, ", ") 
            << std::endl;

    // Start from the first critical node
    // Coarse granularity stage
    //   int coarse_index = 0;
    int MAX_COARSE_ITERATIONS = 500;
    int stagnation_count = 0;  // Track how many iterations passed without improvement

    //   Solution newSol = safestSolution;
    Solution newSol = bestSolution;
    std::unordered_map<int, int> modified_nodes;
    for (int coarseiter = 0; coarseiter < MAX_COARSE_ITERATIONS; ++coarseiter) {
        if (coarseiter >= criticalNodes.size()) break;
        int node_index = criticalNodes[coarseiter];  // Pick the most critical node first
        int pre_option_idx = newSol[node_index];  // Store previous option

        // Set node to its minimum-cost option
        newSol[node_index] = minCostIndices[node_index];
        // modified_nodes[node_index] = pre_option_idx;  // Save for feasibility check


        double cost = ComputeCost(newSol, problem);
        // If feasible and improves cost, update the best solution
        if (IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict, nodeToUnions, problem) && (bestCost - cost > bestCost * 0.001 / std::pow(10,restart_count))) {
            // double cost = ComputeCost(newSol, problem);
            // if () {
                std::cout << "#### WOW ### " << coarseiter + 1 << "\n";
                bestCost = cost;
                bestSolution = newSol;
                stagnation_count = 0;  // Reset stagnation counter

                // Remove node from unchangedCriticalNodes since it was successfully changed
                unchangedCriticalNodes.erase(
                    std::remove(unchangedCriticalNodes.begin(), unchangedCriticalNodes.end(), node_index),
                    unchangedCriticalNodes.end());
            
        } else {
            if (node_index == 14339 or node_index == 19469) {
                    std::cout << "//// 14339 | 19469 ### " << node_index << pre_option_idx << IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict, nodeToUnions, problem) << "\n";
                }
            // Try until last option index
            bool strategy_found = false;
            for (int i = 0; i < problem.nodes[node_index].strategies.size(); ++i){
                newSol[node_index] = i;
                double cost = ComputeCost(newSol, problem);

                if (node_index == 14940) {
                    bool is_feasible = IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict, nodeToUnions, problem);
                    double cost_diff = bestCost - cost;

                    std::cout << "@@ 14940 ### Feasible: " << (is_feasible ? "Yes" : "No")
                            << " | Best Cost: " << bestCost << "opt_idx" << i << "newSol[19929]" << newSol[19929]
                            << " | Cost Difference: " << cost_diff << "\n";
                }

                // double cost = ComputeCost(newSol, problem);
                if (IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict, nodeToUnions, problem) && (bestCost - cost > bestCost * 0.001 / std::pow(10, restart_count))) {
                    // if (bestCost - cost > bestCost * 0.001) {
                        bestCost = cost;
                        bestSolution = newSol;
                        stagnation_count = 0;  // Reset stagnation counter
                        // feasible = false;
                        std::cout << "#### HHHH ### " << coarseiter + 1 << " Node: " << node_index << " succeeded at strategy " << i << "\n";
                        strategy_found = true;

                        // Remove node from unchangedCriticalNodes since it was successfully changed
                        unchangedCriticalNodes.erase(
                            std::remove(unchangedCriticalNodes.begin(), unchangedCriticalNodes.end(), node_index),
                            unchangedCriticalNodes.end());
                        break;  // Exit loop when a feasible strategy is found
                    // }
                }
                
            }
            
            // Revert the node change if infeasible
            if (!strategy_found) {
                std::cout << "#### NOT FOUND ### " << coarseiter + 1 << " Node: " << node_index << "Pre option: " << pre_option_idx << "\n";
                newSol[node_index] = pre_option_idx;
            }
            // break;
        }

        std::cout << "Iteration " << coarseiter + 1 << ": Best Cost = " << bestCost << "\n";

        // Check stagnation condition
        if (++stagnation_count >= 100) {
            std::cout << "#### Stagnation detected after 100 iterations. Recomputing criticality... ####\n";
            break;  // Exit the for-loop and recompute critical nodes
        }
        }
  }
  
  std::cout << "Remaining unchanged critical nodes: "
          << (unchangedCriticalNodes.empty() ? "None" : absl::StrJoin(unchangedCriticalNodes, ", "))
          << std::endl;

  // candidateNodes: Constants for progressive exploration
  constexpr double ratio_interval = 0.1;  // Size of each segment
  constexpr double patience = 1e5;  // Number of iterations before moving to next segment
  constexpr double improvement_threshold = 0.1;  // 1% improvement threshold
  int iteration_since_last_improvement = 0;
  double prev_best_cost = bestCost;

//   int numCandidate = std::max(1, static_cast<int>(ratio_interval * problem.nodes.size()));
//   int numCandidate = std::max(1, 200);
  int numCandidate = std::max(1, static_cast<int>(0.2*problem.nodes.size()));
  int segment_start = 0;  // Start from the top segment
  int segment_end = std::min(segment_start + numCandidate, static_cast<int>(unchangedCriticalNodes.size())); 

//   std::vector<int> candidateNodes(unchangedCriticalNodes.begin() + segment_start, 
//                                 unchangedCriticalNodes.begin() + segment_end);
//   int segment_end = unchangedCriticalNodes.empty() ? 0 
                        // : std::min(segment_start + numCandidate, static_cast<int>(unchangedCriticalNodes.size()));
  
    // Assign an empty vector if unchangedCriticalNodes is empty
  std::vector<int> candidateNodes;
  if (!unchangedCriticalNodes.empty()) {
      candidateNodes.assign(unchangedCriticalNodes.begin() + segment_start, 
                          unchangedCriticalNodes.begin() + segment_end);
  }

  // Optimization parameters.
  const int POP_SIZE = 30;
  const int MAX_ITERATIONS = 0;
  const double ALPHA = 0.5;  // Exploration probability.
  const double BETA  = 0.5;  // Exploitation probability.

  // Initialize the population with the safest solution.
  std::vector<Solution> population(POP_SIZE, safestSolution);
  // Set up a random number generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  // Check if `candidateNodes` is empty before setting up node selection
  std::optional<std::uniform_int_distribution<>> node_selector;
  if (!candidateNodes.empty()) {
         node_selector.emplace(0, candidateNodes.size() - 1);
        //  node_selector(0, candidateNodes.size() - 1);
  }

  // Main SCSO loop.
  for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
    if (absl::Now() - start_time >= timeout) {
      std::cout << "Timeout reached. Returning best solution found.\n";
      return bestSolution;
    }
    std::vector<Solution> newPopulation;
    for (auto &sol : population) {
      Solution newSol = sol;
      std::unordered_map<int, int> modified_nodes;

      if (!candidateNodes.empty() && dis(gen) < ALPHA) {
            // Exploration: randomly modify some nodes
            // int numToModify = std::max(1, static_cast<int>(problem.nodes.size() * 0.001));
            int numToModify = 1;

            for (int i = 0; i < numToModify; ++i) {
                if (node_selector.has_value()) {
                    // int node_index = criticalNodes[node_selector(gen)];
                    // int node_index = candidateNodes[node_selector(gen)];
                    int node_index = candidateNodes[node_selector->operator()(gen)];
                    int pre_option_idx = newSol[node_index];  // Store previous option
                    std::uniform_int_distribution<> strategy_selector(0, problem.nodes[node_index].strategies.size() - 1);
                    newSol[node_index] = strategy_selector(gen);
                    modified_nodes[node_index] = pre_option_idx;  // Save previous choice for feasibility check
                }
            }

        } else {
            // Exploitation: move towards best solution
            std::vector<int> nodeKeys;
            for (size_t i = 0; i < problem.nodes.size(); ++i) {
                if (!problem.nodes[i].strategies.empty()) { // Ensuring valid keys like macro_nodes.keys()
                    nodeKeys.push_back(i);
                }
            }
            
            for (int node : nodeKeys) {
                if (dis(gen) < BETA) {
                    newSol[node] = bestSolution[node];
                }
            }
        }

      // Feasibility check for modified nodes
      bool feasible = true;
      for (const auto &[node_index, pre_option_idx] : modified_nodes) {
          if (!IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict,
                                  nodeToUnions, problem)) {
              feasible = false;
              break;
          }
      }

      if(feasible)
        newPopulation.push_back(newSol);
      else
        newPopulation.push_back(sol);
    }
    // Evaluate the new population.
    for (const auto &sol : newPopulation) {
      double cost = ComputeCost(sol, problem);
      if (cost < bestCost) {
        // if (bestCost - cost > 0.001 * bestCost) {
        bestCost = cost;
        bestSolution = sol;
        iteration_since_last_improvement = 0;  // Reset improvement tracker
      }
    }
    population = newPopulation;
    std::cout << "Iteration " << iter + 1 << ": Best Cost = " << bestCost << "\n";


    // Stop if we've exhausted all segments
    if (segment_start >= problem.nodes.size()) {
        std::cout << "All critical node segments explored. Stopping optimization.\n";
        break;
    }
  }

  return bestSolution;
}

}  // namespace iopddl
