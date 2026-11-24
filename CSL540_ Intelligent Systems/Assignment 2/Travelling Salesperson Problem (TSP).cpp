#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdint>

using namespace std;
using namespace std::chrono;

const int NUM_NODES = 15; 
const double INF = 1e9;

struct State {
    int current_node;
    uint64_t mask;
    double g; 
    double h; 
    double f; 
    vector<int> path;
    
    bool operator>(const State& other) const {
        return f > other.f;
    }
};


double getMSTHeuristic(uint64_t mask, int current_node, const vector<vector<double>>& dist) {
    vector<int> unvisited;
    for (int i = 0; i < NUM_NODES; i++) {
        if (!(mask & (1ULL << i))) unvisited.push_back(i);
    }

    if (unvisited.empty()) return dist[current_node][0];
    if (unvisited.size() == 1) return dist[current_node][unvisited[0]] + dist[unvisited[0]][0];

    double mst_cost = 0;
    vector<bool> in_mst(unvisited.size(), false);
    vector<double> min_edge(unvisited.size(), INF);
    
    in_mst[0] = true;
    for (size_t i = 1; i < unvisited.size(); i++) {
        min_edge[i] = dist[unvisited[0]][unvisited[i]];
    }

    for (size_t i = 1; i < unvisited.size(); i++) {
        int next_node = -1;
        double min_val = INF;
        for (size_t j = 0; j < unvisited.size(); j++) {
            if (!in_mst[j] && min_edge[j] < min_val) {
                min_val = min_edge[j];
                next_node = j;
            }
        }
        mst_cost += min_val;
        in_mst[next_node] = true;
        for (size_t j = 0; j < unvisited.size(); j++) {
            if (!in_mst[j] && dist[unvisited[next_node]][unvisited[j]] < min_edge[j]) {
                min_edge[j] = dist[unvisited[next_node]][unvisited[j]];
            }
        }
    }

    double min_to_mst = INF, min_from_mst = INF;
    for (int u : unvisited) {
        min_to_mst = min(min_to_mst, dist[current_node][u]);
        min_from_mst = min(min_from_mst, dist[u][0]);
    }

    return mst_cost + min_to_mst + min_from_mst;
}

int main() {
    random_device rd;
    mt19937 gen(42); 
    uniform_real_distribution<> dis(10.0, 99.0);

    vector<pair<double, double>> coords(NUM_NODES);
    for (int i = 0; i < NUM_NODES; ++i) coords[i] = {dis(gen), dis(gen)};

    vector<vector<double>> dist(NUM_NODES, vector<double>(NUM_NODES, 0.0));
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < NUM_NODES; ++j) {
            if (i != j) dist[i][j] = hypot(coords[i].first - coords[j].first, coords[i].second - coords[j].second);
        }
    }

    cout << "--- INITIAL ADJACENCY MATRIX (" << NUM_NODES << "x" << NUM_NODES << ") ---\n";
    
    cout << "      ";
    for (int i = 0; i < min(NUM_NODES, 15); i++) {
        cout << setw(7) << i;
    }
    cout << "\n   " << string(min(NUM_NODES, 15) * 7 + 3, '-') << "\n";

    for (int i = 0; i < min(NUM_NODES, 15); i++) {
        cout << setw(2) << i << " | ";
        for (int j = 0; j < min(NUM_NODES, 15); j++) {
            if (i == j) cout << setw(7) << "-";
            else cout << setw(7) << fixed << setprecision(1) << dist[i][j];
        }
        cout << "\n";
    }
    if (NUM_NODES > 15) cout << "(Matrix truncated for readability)\n";
    cout << "\n";

    uint64_t ALL_VISITED = (1ULL << NUM_NODES) - 1;
    priority_queue<State, vector<State>, greater<State>> pq;
    
    double initial_h = getMSTHeuristic(1ULL, 0, dist);
    pq.push({0, 1ULL, 0.0, initial_h, initial_h, {0}});

    long long nodes_expanded = 0;
    
    cout << "Start Node: 0\n\n";

    cout << "--- HEURISTIC TABLE (From Node 0) ---\n";
    cout << setw(10) << "Next Node" << " | " << "MST Heuristic (h)\n";
    cout << "---------------------------------\n";
    for (int i = 1; i < NUM_NODES; i++) {
        uint64_t next_mask = 1ULL | (1ULL << i);
        double h_val = getMSTHeuristic(next_mask, i, dist);
        cout << setw(10) << i << " | " << fixed << setprecision(2) << h_val << "\n";
    }

    cout << "\nSolving...\n";
    
    auto start_time = high_resolution_clock::now();

    while (!pq.empty()) {
        State curr = pq.top();
        pq.pop();
        nodes_expanded++;

        if (curr.mask == ALL_VISITED) {
            auto stop_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop_time - start_time);

            curr.path.push_back(0); 

            cout << "--- SOLUTION FOUND ---\n";
            cout << "Optimal Path Taken: ";
            for (size_t i = 0; i < curr.path.size(); i++) {
                cout << curr.path[i] << (i == curr.path.size() - 1 ? "" : " -> ");
            }
            
            cout << "\n\n--- PATH CALCULATION BREAKDOWN ---\n";
            double check_cost = 0;
            for (size_t i = 0; i < curr.path.size() - 1; i++) {
                int u = curr.path[i];
                int v = curr.path[i+1];
                double edge_cost = dist[u][v];
                check_cost += edge_cost;
                cout << "Edge " << u << " -> " << v << " : " << fixed << setprecision(2) << edge_cost << "\n";
            }
            cout << "----------------------------------\n";
            cout << "Total Minimum Cost : " << fixed << setprecision(2) << check_cost << "\n";

            cout << "\n--- PERFORMANCE METRICS ---\n";
            cout << "Total Nodes Expanded: " << nodes_expanded << "\n";
            cout << "Time Taken: " << duration.count() << " ms\n";
            return 0;
        }

        for (int next_node = 0; next_node < NUM_NODES; next_node++) {
            if (!(curr.mask & (1ULL << next_node))) {
                uint64_t next_mask = curr.mask | (1ULL << next_node);
                double next_g = curr.g + dist[curr.current_node][next_node];
                double h = getMSTHeuristic(next_mask, next_node, dist);
                
                vector<int> next_path = curr.path;
                next_path.push_back(next_node);
                
                pq.push({next_node, next_mask, next_g, h, next_g + h, next_path});
            }
        }
    }
    return 0;
}