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

#include <iostream>
#include <unordered_map>
#include <map>
#include <queue>
#include <limits>

#include "absl/strings/str_join.h"


namespace iopddl {


struct Cluster {
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::vector<std::pair<int, int>>>>> data;
};

#include <functional> // Required for std::hash
std::pair<Cluster, std::set<int>> create_clusters(const Problem &problem) {
// Cluster create_clusters(const Problem &problem) {
    // Define a custom hash function for std::pair<int, int>
    struct hash_pair {
        size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    std::unordered_map<std::pair<int, int>, std::vector<double>, hash_pair> edge_costs;
    std::set<int> big_macro_nodes;
    std::vector<std::pair<int, int>> big_macro_edges;

    // Step 1: Identify "big" macro nodes based on edge cost variation
    for (const auto &edge : problem.edges) {
        int src = edge.nodes[0], dst = edge.nodes[1];
        
        if (edge.strategies.empty()) continue;

        std::vector<double> costs;
        for (const auto &strategy : edge.strategies) {
            costs.push_back(strategy.cost);
        }

        double min_cost = *std::min_element(costs.begin(), costs.end());
        double max_cost = *std::max_element(costs.begin(), costs.end());

        edge_costs[{src, dst}] = costs;

        // if (src == 2477 or dst == 2477){
        //         std::cout << "src " << src << " ; dst " << dst << "max_cost - min_cost" << max_cost - min_cost << std::endl;
        //     }

        if (max_cost - min_cost >= 1e18) {
            big_macro_nodes.insert(src);
            big_macro_nodes.insert(dst);
            big_macro_edges.emplace_back(src, dst);
            if (src == 31376 or dst == 31376 or src == 31385 or dst == 31385){
                std::cout << "Adding " << src << " and " << dst << " to big_macro_nodes." << std::endl;
            }
        }
    }

    if (big_macro_nodes.count(31774) == 0) {  // Only keep nodes NOT in big_macro_nodes
            std::cout << "31774 not in big nodes" << std::endl;
        }
    else {
        std::cout << "31774 in big nodes" << std::endl;
    }


    // Step 2: Cluster "big" macro nodes using BFS
    std::vector<std::set<int>> clusters;
    std::set<int> visited;
    
    for (int node : big_macro_nodes) {
        if (!visited.count(node)) {
            std::queue<int> q;
            std::set<int> cluster;
            
            q.push(node);
            visited.insert(node);

            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                cluster.insert(curr);

                for (const auto &[src, dst] : big_macro_edges) {
                    if (src == curr && big_macro_nodes.count(dst) && !visited.count(dst)) {
                        q.push(dst);
                        visited.insert(dst);
                    }
                    if (dst == curr && big_macro_nodes.count(src) && !visited.count(src)) {
                        q.push(src);
                        visited.insert(src);
                    }
                }
            }
            clusters.push_back(std::move(cluster));
        }
    }


    // Step 3: Store all possible good option pairs per cluster
    Cluster cluster_map;
    
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        const auto &cluster_nodes = clusters[cluster_id];

        // Use a set to store unique [src, strategy_group] keys.
        std::set<std::pair<int, int>> stored_groups;

        for (const auto &[edge, costs] : edge_costs) {
            int src = edge.first, dst = edge.second;

            // if (cluster_nodes.count(src) && cluster_nodes.count(dst) ) {
            if (cluster_nodes.count(src) && cluster_nodes.count(dst) && std::find(big_macro_edges.begin(), big_macro_edges.end(), std::make_pair(src, dst)) != big_macro_edges.end()) {
                size_t num_dst_strategies = problem.nodes[dst].strategies.size();
                
                for (size_t option_idx = 0; option_idx < costs.size(); ++option_idx) {
                    if (costs[option_idx] < 1e17) {
                        int strategy_group = option_idx / num_dst_strategies;
                        // int assigned_option = option_idx % num_dst_strategies;
                        
                        // // Debugging: Print cluster details if src == 2477
                        // if (src == 31770) {
                        //     std::cout << "Cluster ID: " << cluster_id
                        //                 << " | src: " << src
                        //                 << " | dst: " << dst
                        //                 << " | option_idx: " << option_idx
                        //                 << " | strategy_group: " << (option_idx / num_dst_strategies)
                        //                 << " | assigned_option: " << (option_idx % num_dst_strategies)
                        //                 << " | cost: " << costs[option_idx]
                        //                 << std::endl;
                        // }

                        cluster_map.data[cluster_id][src][option_idx / num_dst_strategies]
                            .emplace_back(dst, option_idx % num_dst_strategies);

                        // // Check if [dst, option_idx] is already in stored_groups
                        // for (const auto &[stored_src, stored_group] : stored_groups) {
                        //     if (stored_src == dst and stored_group == assigned_option) {
                        //         std::cout << "cluster id " << cluster_id << "stored_src" << stored_src << "stored_group" << stored_group << std::endl;
                        //         // Append previous mappings to the current list
                        //         cluster_map.data[cluster_id][src][strategy_group].insert(
                        //             cluster_map.data[cluster_id][src][strategy_group].end(),
                        //             cluster_map.data[cluster_id][stored_src][stored_group].begin(),
                        //             cluster_map.data[cluster_id][stored_src][stored_group].end()
                        //         );
                        //     }
                        // }
                        stored_groups.insert({src, strategy_group});
                        // if (cluster_id == 11) {
                        //     // Print all elements in stored_groups after insertion
                        // std::cout << "Stored Groups after insertion: ";
                        // for (const auto &[stored_src, stored_group] : stored_groups) {
                        //     std::cout << "[" << stored_src << ", " << stored_group << "] ";
                        // }
                        // std::cout << std::endl;
                        // }
                    }
                }
            }
        }
        for (const auto &[edge, costs] : edge_costs) {
            int src = edge.first, dst = edge.second;

            // if (cluster_nodes.count(src) && cluster_nodes.count(dst) ) {
            if (cluster_nodes.count(src) && cluster_nodes.count(dst) && std::find(big_macro_edges.begin(), big_macro_edges.end(), std::make_pair(src, dst)) != big_macro_edges.end()) {
                size_t num_dst_strategies = problem.nodes[dst].strategies.size();
                
                for (size_t option_idx = 0; option_idx < costs.size(); ++option_idx) {
                    if (costs[option_idx] < 1e17) {
                        int strategy_group = option_idx / num_dst_strategies;
                        int assigned_option = option_idx % num_dst_strategies;
                        
                        // // Debugging: Print cluster details if src == 2477
                        // if (src == 31770) {
                        //     std::cout << "Cluster ID: " << cluster_id
                        //                 << " | src: " << src
                        //                 << " | dst: " << dst
                        //                 << " | option_idx: " << option_idx
                        //                 << " | strategy_group: " << (option_idx / num_dst_strategies)
                        //                 << " | assigned_option: " << (option_idx % num_dst_strategies)
                        //                 << " | cost: " << costs[option_idx]
                        //                 << std::endl;
                        // }

                        // cluster_map.data[cluster_id][src][option_idx / num_dst_strategies]
                        //     .emplace_back(dst, option_idx % num_dst_strategies);

                        // Check if [dst, option_idx] is already in stored_groups
                        for (const auto &[stored_src, stored_group] : stored_groups) {
                            if (stored_src == dst and stored_group == assigned_option) {
                                // std::cout << "cluster id " << cluster_id << "stored_src" << stored_src << "stored_group" << stored_group << std::endl;
                                // Append previous mappings to the current list

                                // auto &stored_list = cluster_map.data[cluster_id][stored_src][stored_group];
                                // if (std::find(stored_list.begin(), stored_list.end(), std::make_pair(31385, 0)) != stored_list.end()) {
                                //     std::cout << "Found [31385, 0] in cluster_map.data[" << cluster_id 
                                //             << "][" << stored_src << "][" << stored_group << "]\n";
                                // }

                                cluster_map.data[cluster_id][src][strategy_group].insert(
                                    cluster_map.data[cluster_id][src][strategy_group].end(),
                                    cluster_map.data[cluster_id][stored_src][stored_group].begin(),
                                    cluster_map.data[cluster_id][stored_src][stored_group].end()
                                );
                            }
                        }
                    }
                }
            }
        }

    }
    

    // Debug print if a cluster has multiple src values.
    // if (cluster_map.data[11].size() > 1) {
    //     std::cout << "Cluster ID: 11 has multiple src values:" << std::endl;
    //     for (const auto& [src, strategy_groups] : cluster_map.data[11]) {
    //         std::cout << "  src: " << src << std::endl;
    //         for (const auto& [strategy_group, pairs] : strategy_groups) {
    //             std::cout << "    Strategy Group: " << strategy_group << std::endl;
    //             for (const auto& [dst, assigned_option] : pairs) {
    //                 std::cout << "      dst: " << dst << ", assigned_option: " << assigned_option << std::endl;
    //             }
    //         }
    //     }
    // }
    
    // return cluster_map;
    return {cluster_map, big_macro_nodes};
}



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

std::unordered_map<int, double> CheckSafestSolutionFeasible(
    const Solution &solution,
    const std::vector<std::set<int>> &allUnions,
    const Problem &problem) {
  
  if (!problem.usage_limit)
    return {};  // No limit means always feasible.

  double limit = *problem.usage_limit;
  std::unordered_map<int, double> union_mem_dict;

  for (size_t uidx = 0; uidx < allUnions.size(); ++uidx) {
    double totalUsage = 0;
    for (int node : allUnions[uidx]) {
      totalUsage += problem.nodes[node].strategies[solution.at(node)].usage;
    //   if (totalUsage > limit) {
    //     std::cout << " >>> uidx: <<< " << uidx << "; usage: " << problem.nodes[node].strategies[solution.at(node)].usage
    //         << std::endl;
    //   }
    }
    if (totalUsage > limit) {
      std::cout << " >>> infeasible union <<< " 
            << absl::StrJoin(allUnions[uidx], ", ") 
            << std::endl;
      std::cout << "uid" << uidx << " >>> totalUsage: <<< " << totalUsage << "; limit: " << limit
            << std::endl;
      return {};  // Indicates infeasible solution
    }
    union_mem_dict[uidx] = totalUsage;
  }
  
  return union_mem_dict;
}

bool IsFeasibleGroupUpdated(
    const std::unordered_map<int, int> &modified_nodes,
    Solution &newSol,  // Ensure newSol contains updated options
    std::unordered_map<int, double> &union_mem_dict,
    const std::unordered_map<int, std::vector<int>> &node_to_unions,
    const Problem &problem) {

    if (!problem.usage_limit)
        return true;

    double limit = *problem.usage_limit;
    std::unordered_map<int, double> temp_union_mem_dict = union_mem_dict;  // Copy for tentative update

    // **Step 1: First, subtract all previous usages**
    for (const auto &[node_index, pre_option_idx] : modified_nodes) {
        if (node_to_unions.find(node_index) == node_to_unions.end()) {
            continue;  // Node does not belong to any union
        }

        const auto &macro_usages = problem.nodes[node_index].strategies;

        for (int uindex : node_to_unions.at(node_index)) {
            temp_union_mem_dict[uindex] -= macro_usages[pre_option_idx].usage;
        }
    }

    // **Step 2: Then, add new usages one-by-one and check feasibility**
    for (const auto &[node_index, _] : modified_nodes) {  // `_` because modified_nodes contains pre_option_idx
        const auto &macro_usages = problem.nodes[node_index].strategies;
        int new_option_idx = newSol[node_index];  // Get the new option

        for (int uindex : node_to_unions.at(node_index)) {
            double cur_mem_usage = temp_union_mem_dict[uindex] + macro_usages[new_option_idx].usage;

            if (cur_mem_usage > limit) {
                return false;  // Reject entire modification batch
            }

            temp_union_mem_dict[uindex] = cur_mem_usage;  // Apply the new usage update
        }
    }

    // **Step 3: If all modifications are feasible, commit them**
    // for (const auto &[node_index, _] : modified_nodes) {
    //     solution[node_index] = newSol[node_index];  // Apply new options
    // }
    union_mem_dict = temp_union_mem_dict;

    return true;
}



// Function to check if modifying a node maintains feasibility
bool IsFeasibleUpdated(
    int node_index,
    int pre_option_idx,
    Solution &solution,
    std::unordered_map<int, double> &union_mem_dict,
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
  double limit = *problem.usage_limit;
  for (const auto &unionSet : parentUnions) {
    double totalUsage = 0;
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
    const Problem &problem,
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


    if (CurOptCost >= 1e12){
        sigcrit[i] = 1.0;
        // crit[i] += 0.0001;
        std::cout << "node: " << i << "curOptCost" << CurOptCost << "mincost " << minCost << "cur opt:" << safestSolution[i] <<std::endl; 
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
  std::unordered_map<int, double> union_mem_dict = CheckSafestSolutionFeasible(safestSolution, allUnions, problem);

  if (union_mem_dict.empty()) {
      std::cout << "The safest solution violates the memory usage limit in some parent union!" << std::endl;
      throw std::runtime_error("Safest solution is not feasible");
  }
  

//   Solution newSol = safestSolution;
  Solution newSol = bestSolution;
  std::unordered_map<int, int> modified_nodes;



  // Main SCSO loop.
  // Step 1: Create clusters
    // Cluster big_nodes_clusters = create_clusters(problem);
    auto [big_nodes_clusters, big_macro_nodes] = create_clusters(problem);


    // Step 2: Group Modify Loop
    // Solution newSol = bestSolution;  // Copy best solution
    for (const auto &[cluster_id, cluster_info] : big_nodes_clusters.data) {
        std::unordered_map<int, int> modified_nodes;
        bool feasible_flag = true;
        newSol = bestSolution;  // Reset solution for each cluster

        for (const auto &[start_node_index, candidate_options_info] : cluster_info) {
            
            // modified_nodes.clear();
            feasible_flag = true;
            newSol = bestSolution;  // Reset solution for each node
            int node_index = start_node_index;

            for (const auto &[cur_option_index, adjacent_nodes_info] : candidate_options_info) {
                feasible_flag = true;
                newSol = bestSolution;  // Reset solution for each option change
                modified_nodes.clear();

                // Change main node option
                int pre_option_idx = bestSolution[node_index];
                newSol[node_index] = cur_option_index;
                modified_nodes[node_index] = pre_option_idx;

                // Feasibility check for the changed node
                // if (!IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict, nodeToUnions, problem)) {
                //     feasible_flag = false;
                //     if (node_index == 2477) {std::cout << "infeasible node: " << node_index << "pre_option_idx" << pre_option_idx << "cur_option_index" << cur_option_index << std::endl; }
                //     // std::cout << "infeasible node: " << node_index << "pre_option_idx" << pre_option_idx << "cur_option_index" << cur_option_index << std::endl; 
                //     continue;
                // }

                // else{
                //     std::cout << "feasible node: " << node_index << "pre_option_idx" << pre_option_idx << "cur_option_index" << cur_option_index << std::endl; 
                // }

                // Modify adjacent nodes in the cluster
                for (const auto &[adjacent_node_index, adj_option_idx] : adjacent_nodes_info) {
                    int pre_option_idx_adj = bestSolution[adjacent_node_index];
                    newSol[adjacent_node_index] = adj_option_idx;
                    modified_nodes[adjacent_node_index] = pre_option_idx_adj;

                    // if (!IsFeasibleUpdated(adjacent_node_index, pre_option_idx_adj, newSol, union_mem_dict, nodeToUnions, problem)) {
                    //     feasible_flag = false;
                    if (node_index == 1250) {std::cout << "adjacent_node_index" << adjacent_node_index << "pre_option_idx_adj" << pre_option_idx_adj << "adj_option_idx" << adj_option_idx << std::endl; }
                    //     // std::cout << "infea node: " << adjacent_node_index << "pre_option_idx_adj" << pre_option_idx_adj << "adj_option_idx" << adj_option_idx << std::endl; 
                    //     break;
                    }

                if (!IsFeasibleGroupUpdated(modified_nodes, newSol, union_mem_dict, nodeToUnions, problem)) {
                    feasible_flag = false;
                    // std::cout << "Cluster " << cluster_id << "start_node_index" << start_node_index << "cur_option_index" << cur_option_index << "bestSolution[node_index]" << bestSolution[node_index] << "adjacent_nodes_info.size()" << adjacent_nodes_info.size() << " is infeasible with given modifications." << std::endl;
                }
                // else {
                //     std::cout << "OK Cluster " << cluster_id << "start_node_index" << start_node_index << "cur_option_index" << cur_option_index << "bestSolution[node_index]" << bestSolution[node_index] << "adjacent_nodes_info.size()" << adjacent_nodes_info.size() << "feasible_flag" << feasible_flag << std::endl;
                // }

                // If modification is feasible, check for cost improvement
                if (feasible_flag) {
                    double new_cost = ComputeCost(newSol, problem);
                    // if (bestCost - new_cost > 0.001 * bestCost) {
                    if (bestCost - new_cost > 1e5) {
                        bestSolution = newSol;
                        bestCost = new_cost;
                        std::cout << "Updated Best Cost: " << bestCost
                                    << " | Node: " << start_node_index
                                    << " | New Option: " << cur_option_index
                                    << " | Previous Option: " << pre_option_idx
                                    << " | Adjacent Nodes: " << adjacent_nodes_info.size()
                                    << std::endl;
                    }
                    // else{
                    //     std::cout << "start node index" << start_node_index << "diff" << bestCost - new_cost << std::endl; 
                    // }
                }

                //     else{
                //     std::cout << "feasible node: " << adjacent_node_index << "pre_option_idx_adj" << pre_option_idx_adj << "adj_option_idx" << adj_option_idx << std::endl;
                // }
                }
                
            }
    // }

    // // Main SCSO loop.
    // // Step 1: Create clusters
    // // Cluster big_nodes_clusters = create_clusters(problem);
    // auto [big_nodes_clusters, big_macro_nodes] = create_clusters(problem);


    // // Step 2: Group Modify Loop
    // // Solution newSol = bestSolution;  // Copy best solution
    // for (const auto &[cluster_id, cluster_info] : big_nodes_clusters.data) {
    //     std::unordered_map<int, int> modified_nodes;
    //     bool feasible_flag = true;
    //     newSol = bestSolution;  // Reset solution for each cluster

    //     for (const auto &[start_node_index, candidate_options_info] : cluster_info) {
            
    //         modified_nodes.clear();
    //         feasible_flag = true;
    //         newSol = bestSolution;  // Reset solution for each node
    //         int node_index = start_node_index;

    //         for (const auto &[cur_option_index, adjacent_nodes_info] : candidate_options_info) {
    //             feasible_flag = true;
    //             newSol = bestSolution;  // Reset solution for each option change
    //             modified_nodes.clear();

    //             // Change main node option
    //             int pre_option_idx = bestSolution[node_index];
    //             newSol[node_index] = cur_option_index;
    //             modified_nodes[node_index] = pre_option_idx;

    //             // Feasibility check for the changed node
    //             if (!IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict, nodeToUnions, problem)) {
    //                 feasible_flag = false;
    //                 if (node_index == 2477) {std::cout << "infeasible node: " << node_index << "pre_option_idx" << pre_option_idx << "cur_option_index" << cur_option_index << std::endl; }
    //                 // std::cout << "infeasible node: " << node_index << "pre_option_idx" << pre_option_idx << "cur_option_index" << cur_option_index << std::endl; 
    //                 continue;
    //             }

    //             // else{
    //             //     std::cout << "feasible node: " << node_index << "pre_option_idx" << pre_option_idx << "cur_option_index" << cur_option_index << std::endl; 
    //             // }

    //             // Modify adjacent nodes in the cluster
    //             for (const auto &[adjacent_node_index, adj_option_idx] : adjacent_nodes_info) {
    //                 int pre_option_idx_adj = bestSolution[adjacent_node_index];
    //                 newSol[adjacent_node_index] = adj_option_idx;
    //                 modified_nodes[adjacent_node_index] = pre_option_idx_adj;

    //                 if (!IsFeasibleUpdated(adjacent_node_index, pre_option_idx_adj, newSol, union_mem_dict, nodeToUnions, problem)) {
    //                     feasible_flag = false;
    //                     if (node_index == 2477) {std::cout << "infea node: " << adjacent_node_index << "pre_option_idx_adj" << pre_option_idx_adj << "adj_option_idx" << adj_option_idx << std::endl; }
    //                     // std::cout << "infea node: " << adjacent_node_index << "pre_option_idx_adj" << pre_option_idx_adj << "adj_option_idx" << adj_option_idx << std::endl; 
    //                     break;
    //                 }
    //             //     else{
    //             //     std::cout << "feasible node: " << adjacent_node_index << "pre_option_idx_adj" << pre_option_idx_adj << "adj_option_idx" << adj_option_idx << std::endl;
    //             // }
    //             }

    //             // If modification is feasible, check for cost improvement
    //             if (feasible_flag) {
    //                 double new_cost = ComputeCost(newSol, problem);
    //                 // if (bestCost - new_cost > 0.001 * bestCost) {
    //                 if (bestCost - new_cost > 1e6) {
    //                     bestSolution = newSol;
    //                     bestCost = new_cost;
    //                     std::cout << "Updated Best Cost: " << bestCost
    //                                 << " | Node: " << start_node_index
    //                                 << " | New Option: " << cur_option_index
    //                                 << " | Previous Option: " << pre_option_idx
    //                                 << " | Adjacent Nodes: " << adjacent_nodes_info.size()
    //                                 << std::endl;
    //                 }
    //                 // else{
    //                 //     std::cout << "start node index" << start_node_index << "diff" << bestCost - new_cost << std::endl; 
    //                 // }
    //             }
    //         }
    //     }
    }

    auto [critScores, sigcrit, minCostIndices] = ComputeCriticality(problem, bestSolution, bestCost);  
    std::vector<int> sortedNodes = allNodes;
    std::sort(sortedNodes.begin(), sortedNodes.end(), [&](int a, int b) {
        return critScores[a] > critScores[b];
    });

    std::vector<int> top_critical_nodes;
    for (int node : sortedNodes) {
        if (big_macro_nodes.count(node) == 0) {  // Only keep nodes NOT in big_macro_nodes
            top_critical_nodes.push_back(node);
        }
        // if (top_critical_nodes.size() >= 100) break;  // Limit to top 100 nodes (adjustable)
    }

    std::cout << "Number of top critical nodes (excluding big_macro_nodes): " << top_critical_nodes.size() << std::endl;


    // Step 5: Modify loop to process top_critical_nodes first
    for (int node_index : top_critical_nodes) {
        std::unordered_map<int, int> modified_nodes;
        bool feasible_flag = true;
        newSol = bestSolution;  // Reset solution for each critical node

        // for (size_t option_idx = 0; option_idx < problem.nodes[node_index].strategies.size(); ++option_idx) {
        feasible_flag = true;
        newSol = bestSolution;
        modified_nodes.clear();

        // Change main node option
        int pre_option_idx = bestSolution[node_index];
        newSol[node_index] = minCostIndices[node_index];
        modified_nodes[node_index] = pre_option_idx;

        // Feasibility check
        if (!IsFeasibleUpdated(node_index, pre_option_idx, newSol, union_mem_dict, nodeToUnions, problem)) {
            feasible_flag = false;
            continue;
        }

        // Compute cost improvement
        double new_cost = ComputeCost(newSol, problem);

        
        if (bestCost - new_cost > 0.0005 * bestCost) {
            bestSolution = newSol;
            bestCost = new_cost;
            std::cout << "Criticality-based Updated Best Cost: " << bestCost
                    << " | Node: " << node_index
                    << " | New Option: " << minCostIndices[node_index]
                    << " | Previous Option: " << pre_option_idx
                    << std::endl;
        }
        // }
    }



    return bestSolution;

//   return bestSolution;
}

}  // namespace iopddl
