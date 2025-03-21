#include "iopddl_solver.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <bitset>

using namespace std;
using json = nlohmann::json;

// Parse JSON file into structured ProblemData
ProblemData parse_json_input(const std::string& filename) {
    ifstream file(filename);
    json data;
    file >> data;

    ProblemData problem;
    for (const auto& interval : data["problem"]["nodes"]["intervals"]) {
        problem.node_intervals.emplace_back(interval[0], interval[1]);
    }
    
    for (size_t i = 0; i < data["problem"]["nodes"]["costs"].size(); i++) {
        problem.node_costs[i] = data["problem"]["nodes"]["costs"][i].get<vector<int>>();
        problem.node_usages[i] = data["problem"]["nodes"]["usages"][i].get<vector<int>>();
    }

    problem.usage_limit = data["problem"]["usage_limit"];

    for (size_t i = 0; i < data["problem"]["edges"]["nodes"].size(); i++) {
        int src = data["problem"]["edges"]["nodes"][i][0];
        int dest = data["problem"]["edges"]["nodes"][i][1];
        int num_subnodes_src = problem.node_costs[src].size();
        int num_subnodes_dest = problem.node_costs[dest].size();

        for (int p = 0; p < num_subnodes_src; p++) {
            for (int q = 0; q < num_subnodes_dest; q++) {
                problem.edges[{src, p, dest, q}] = data["problem"]["edges"]["costs"][i][p * num_subnodes_dest + q];
            }
        }
    }

    return problem;
}

// Sweep-line algorithm for computing parent unions
std::set<std::set<int>> compute_parent_unions(const ProblemData& problem) {
    vector<tuple<int, char, int>> events;

    for (size_t i = 0; i < problem.node_intervals.size(); i++) {
        int start = problem.node_intervals[i].first;
        int end = problem.node_intervals[i].second;
        if (start != end) {
            events.emplace_back(start, 's', i);
            events.emplace_back(end, 'e', i);
        }
    }

    sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        return get<0>(a) < get<0>(b) || (get<0>(a) == get<0>(b) && get<1>(a) == 'e');
    });

    set<int> active_nodes;
    set<set<int>> parent_unions;
    
    for (const auto& [time, type, node] : events) {
        if (!active_nodes.empty()) {
            parent_unions.insert(active_nodes);
        }
        if (type == 's') {
            active_nodes.insert(node);
        } else {
            active_nodes.erase(node);
        }
    }

    return parent_unions;
}

// Check feasibility
bool is_feasible(const unordered_map<int, int>& solution, const set<set<int>>& parent_unions, 
                 const unordered_map<int, vector<int>>& macro_usages, int usage_limit) {
    for (const auto& union_set : parent_unions) {
        int total_usage = 0;
        for (int i : union_set) {
            total_usage += macro_usages.at(i)[solution.at(i)];
        }
        if (total_usage > usage_limit) {
            return false;
        }
    }
    return true;
}

// Compute total cost
int compute_cost(const unordered_map<int, int>& solution, const unordered_map<int, vector<int>>& macro_nodes, 
                 const unordered_map<tuple<int, int, int, int>, int>& edges) {
    int total_cost = 0;
    for (const auto& [node, opt] : solution) {
        total_cost += macro_nodes.at(node)[opt];
    }
    for (const auto& [edge, cost] : edges) {
        int src = get<0>(edge), p = get<1>(edge), dest = get<2>(edge), q = get<3>(edge);
        if (solution.at(src) == p && solution.at(dest) == q) {
            total_cost += cost;
        }
    }
    return total_cost;
}

// Optimization routine
unordered_map<int, int> optimize_solution(ProblemData& problem, int iterations, int pop_size) {
    unordered_map<int, int> best_solution;
    int best_cost = INT_MAX;

    vector<unordered_map<int, int>> population(pop_size);
    for (auto& sol : population) {
        for (const auto& [i, costs] : problem.node_costs) {
            sol[i] = min_element(costs.begin(), costs.end()) - costs.begin();
        }
    }

    for (int iter = 0; iter < iterations; iter++) {
        for (auto& sol : population) {
            int cost = compute_cost(sol, problem.node_costs, problem.edges);
            if (cost < best_cost && is_feasible(sol, compute_parent_unions(problem), problem.node_usages, problem.usage_limit)) {
                best_solution = sol;
                best_cost = cost;
            }
        }
    }

    return best_solution;
}
