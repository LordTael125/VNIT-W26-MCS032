#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>

using namespace std;
using namespace std::chrono;


const int NUM_NODES = 160;           
const int POP_SIZE = 200;           
const int GENERATIONS = 1500;       
const double MUTATION_RATE = 0.20; 
const int TOURNAMENT_SIZE = 5;      
const int ELITISM_COUNT = 10;       

struct Individual {
    vector<int> chromosome; 
    double cost;            
    double fitness;         

    bool operator<(const Individual& other) const {
        return cost < other.cost; 
    }
};


double evaluateTour(const vector<int>& path, const vector<vector<double>>& dist) {
    double total = 0.0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        total += dist[path[i]][path[i + 1]];
    }
    total += dist[path.back()][path[0]]; 
    return total;
}


const Individual& tournamentSelection(const vector<Individual>& pop, mt19937& gen) {
    uniform_int_distribution<int> dis(0, POP_SIZE - 1);
    int best_idx = dis(gen);
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        int contender_idx = dis(gen);
        if (pop[contender_idx].cost < pop[best_idx].cost) {
            best_idx = contender_idx;
        }
    }
    return pop[best_idx];
}


vector<int> orderedCrossover(const vector<int>& p1, const vector<int>& p2, mt19937& gen) {
    int size = p1.size();
    vector<int> child(size, -1);
    vector<bool> visited(size, false);

   
    child[0] = 0;
    visited[0] = true;

  [  uniform_int_distribution<int> dis(1, size - 1);
    int start = dis(gen);
    int end = dis(gen);
    if (start > end) swap(start, end);

   
    for (int i = start; i <= end; ++i) {
        childi] = p1[i];
        visited[p1[i]] = true;
    }

  
    int curr_pos = (end + 1) % size;
    if (curr_pos == 0) curr_pos = 1;

    for (int i = 1; i < size; ++i) {
        int candidate = p2[i];
        if (!visited[candidate]) {
            child[curr_pos] = candidate;
            visited[candidate] = true;
            curr_pos = (curr_pos + 1) % size;
            if (curr_pos == 0) curr_pos = 1;
        }
    }
    return child;
}


void mutateInversion(vector<int>& chromosome, mt19937& gen) {
    uniform_int_distribution<int> dis(1, NUM_NODES - 1);
    int i = dis(gen);
    int j = dis(gen);
    if (i > j) swap(i, j);
    reverse(chromosome.begin() + i, chromosome.begin() + j + 1);
}

int main() {
    
    mt19937 gen(42);
    uniform_real_distribution<double> coord_dis(10.0, 990.0);

    vector<pair<double, double>> coords(NUM_NODES);
    for (int i = 0; i < NUM_NODES; ++i) {
        coords[i] = {coord_dis(gen), coord_dis(gen)};
    }

   
    vector<vector<double>> dist(NUM_NODES, vector<double>(NUM_NODES, 0.0));
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < NUM_NODES; ++j) {
            if (i != j) {
                dist[i][j] = hypot(coords[i].first - coords[j].first,
                                   coords[i].second - coords[j].second);
            }
        }
    }

   
    int print_limit = min(NUM_NODES, 12);
    cout << "=========================================================\n";
    cout << "   ADJACENCY MATRIX (Showing first " << print_limit << "x" << print_limit << " of " << NUM_NODES << " nodes)\n";
    cout << "=========================================================\n";
    cout << "     ";
    for (int i = 0; i < print_limit; ++i) cout << setw(7) << i;
    cout << "\n   " << string(print_limit * 7 + 5, '-') << "\n";

    for (int i = 0; i < print_limit; ++i) {
        cout << setw(2) << i << " | ";
        for (int j = 0; j < print_limit; ++j) {
            if (i == j) cout << setw(7) << "-";
            else cout << setw(7) << fixed << setprecision(1) << dist[i][j];
        }
        cout << "\n";
    }
    cout << "(Matrix truncated for terminal readability...)\n\n";

    cout << "=========================================================\n";
    cout << "   GENETIC ALGORITHM TSP SOLVER (" << NUM_NODES << " NODES)\n";
    cout << "=========================================================\n";
    cout << "Population Size : " << POP_SIZE << "\n";
    cout << "Generations     : " << GENERATIONS << "\n";
    cout << "Mutation Rate   : " << (MUTATION_RATE * 100) << "% (2-Opt Inversion)\n";
    cout << "Crossover Type  : Ordered Crossover (OX1)\n";
    cout << "---------------------------------------------------------\n";

    auto start_time = high_resolution_clock::now();

   
    vector<Individual> population(POP_SIZE);
    vector<int> base_tour(NUM_NODES);
    iota(base_tour.begin(), base_tour.end(), 0); 

    for (int i = 0; i < POP_SIZE; ++i) {
        population[i].chromosome = base_tour;
       
        shuffle(population[i].chromosome.begin() + 1, population[i].chromosome.end(), gen);
        population[i].cost = evaluateTour(population[i].chromosome, dist);
        population[i].fitness = 1.0 / population[i].cost;
    }

    sort(population.begin(), population.end());
    double initial_best_cost = population[0].cost;
    cout << "Generation 0000 | Initial Best Cost : " << fixed << setprecision(2) << initial_best_cost << "\n";

    uniform_real_distribution<double> prob_dis(0.0, 1.0);

   
    for (int g = 1; g <= GENERATIONS; ++g) {
        vector<Individual> next_gen(POP_SIZE);

       
        for (int i = 0; i < ELITISM_COUNT; ++i) {
            next_gen[i] = population[i];
        }

      
        for (int i = ELITISM_COUNT; i < POP_SIZE; ++i) {
            const Individual& parent1 = tournamentSelection(population, gen);
            const Individual& parent2 = tournamentSelection(population, gen);

            vector<int> child_chrom = orderedCrossover(parent1.chromosome, parent2.chromosome, gen);

            if (prob_dis(gen) < MUTATION_RATE) {
                mutateInversion(child_chrom, gen);
            }

            double child_cost = evaluateTour(child_chrom, dist);
            next_gen[i] = {move(child_chrom), child_cost, 1.0 / child_cost};
        }

        population = move(next_gen);
        sort(population.begin(), population.end());

        
        if (g % 250 == 0 || g == GENERATIONS) {
            cout << "Generation " << setw(4) << setfill('0') << g 
                 << " | Best Tour Cost    : " << fixed << setprecision(2) << population[0].cost << "\n";
        }
    }

    auto stop_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_time - start_time);

  
    const Individual& best = population[0];
    cout << "---------------------------------------------------------\n";
    cout << "SOLUTION CONVERGED!\n";
    cout << "Best Tour Path:\n";
    for (int i = 0; i < NUM_NODES; ++i) {
        cout << best.chromosome[i] << " -> ";
        if ((i + 1) % 12 == 0) cout << "\n";
    }
    cout << "0 (Origin)\n\n";

    double improvement = ((initial_best_cost - best.cost) / initial_best_cost) * 100.0;
    cout << "--- PERFORMANCE METRICS ---\n";
    cout << "Initial Random Best Cost : " << fixed << setprecision(2) << initial_best_cost << "\n";
    cout << "Final Optimized Cost     : " << fixed << setprecision(2) << best.cost << "\n";
    cout << "Cost Reduction           : " << fixed << setprecision(1) << improvement << "%\n";
    cout << "Execution Time           : " << duration.count() << " ms\n";
    cout << "=========================================================\n";

    return 0;
}