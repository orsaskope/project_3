#ifndef HYPERCUBE_HPP
#define HYPERCUBE_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <string>
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <queue>
#include <bitset>

class Hypercube {
public:
    struct VertexResult {
        int index;                 // index of the vector in dataset
        std::string vertex_key;    // binary string key
    };

    // Hypercube parameters
    int kproj;       // number of projections / bits
    int dim;         // input dimension
    double w;        // bucket width for projection
    unsigned seed;   // random seed
    int M;           // max candidate points to check
    int probes;      // max number of vertices to probe

    // Random projections
    std::vector<std::vector<double>> a;  // [kproj][dim]
    std::vector<double> b;               // [kproj]
    
    std::vector<int> r;
    std::vector<int> t;
    std::vector<std::unordered_map<int64_t, int>> hto10;  // stores f(h) for each hash function

    // Hash table: This is where we are going to map the binary keys
    std::unordered_map<std::string, std::vector<VertexResult>> hypercubeTable;

    // Constructor
    Hypercube(int kproj_, int dim_, double w_ = 4.0, unsigned seed_ = 1, int M_ = 10, int probes_ = 2);

    // Projection and vertex computation
    Hypercube::VertexResult compute_vertex(const std::vector<double>& point,int index);

    // query function
    std::vector<Hypercube::VertexResult> query(const std::vector<double>& point, int M);

    // Insert a point into hypercube
    void insert(int index, const std::vector<double>& point);

    // Print stats
    void print_table_info();

    // functions to return top neighbors of query vector
    std::vector<Hypercube::VertexResult> find_top_n_neighbors(
    const std::vector<double>& query,
    const std::vector<VertexResult>& candidates,
    int N,
    const std::vector<std::vector<double>>& dataset);

    // mnist data range search
    std::vector<Hypercube::VertexResult> range_search(
    const std::vector<double>& query_point,
    int radius,
    int M,
    const std::vector<std::vector<double>>& dataset);

    // sift data range search
    std::vector<Hypercube::VertexResult> range_search_sift(
        const std::vector<double>& query_point,
        int radius,
        int M,
        const std::vector<std::vector<double>>& dataset);

private:
    std::mt19937_64 rng;  // random number generator
};

#endif
