#ifndef IOPDDL_SOLVER_H
#define IOPDDL_SOLVER_H

#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <bitset>
#include "json.hpp"  // nlohmann/json library for JSON parsing

using json = nlohmann::json;

// Define structures for problem representation
struct ProblemData {
    std::vector<std::pair<int, int>> node_intervals;  // Start-End time pairs
    std::unordered_map<int, std::vector<int>> node_costs;  // Macro-node index → cost array
    std::unordered_map<int, std::vector<int>> node_usages; // Macro-node index → usage array
    std::unordered_map<std::tuple<int, int, int, int>, int> edges; // (src, p, dest, q) → cost
    int usage_limit;
};

// Core functions
std::set<std::set<int>> compute_parent_unions(const ProblemData& problem);
bool is_feasible(const std::unordered_map<int, int>& solution, 
                 const std::set<std::set<int>>& parent_unions, 
                 const std::unordered_map<int, std::vector<int>>& macro_usages, 
                 int usage_limit);

int compute_cost(const std::unordered_map<int, int>& solution, 
                 const std::unordered_map<int, std::vector<int>>& macro_nodes, 
                 const std::unordered_map<std::tuple<int, int, int, int>, int>& edges);

std::unordered_map<int, int> optimize_solution(ProblemData& problem, int iterations, int pop_size);

// Utility function
ProblemData parse_json_input(const std::string& filename);

#endif // IOPDDL_SOLVER_H