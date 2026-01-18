#ifndef LSH_HPP
#define LSH_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <string>
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>




// === Locality Sensitive Hashing (LSH) class ===
class LSH {
public:

    struct GResult {
        int index;                 // index of the vector in dataset
        std::string g_key;         // string key (for reference)
        unsigned long long ID;     // locality-sensitive ID
    };


    int L;   // number of hash tables
    int k;   // number of hash functions per table
    int dim; // dimension of data
    double w; // bucket width
    unsigned seed;

    // Random parameters
    std::vector<std::vector<std::vector<double>>> a; // [L][k][dim]
    std::vector<std::vector<double>> b;              // [L][k]

    // In LSH.h
    std::vector<std::vector<long long>> r;  // L x k random integer coefficients


    // Hash tables
    std::vector<std::unordered_map<std::string, std::vector<GResult>>> hashTables;


    // Constructor
    LSH(int L_, int k_, int dim_, double w_, unsigned seed_ = 1);

    // Hashing functions
    std::vector<long long> compute_h(const std::vector<double>& point, int tableIdx );

    // g function
    GResult compute_g(const std::vector<double>& point, int tableIdx, int n, int index);

    // insert functions
    void insert(int index, const std::vector<double>& point , int n);

    void print_tables();

    //function to get candidates from lsh hashing
    std::vector<GResult> get_candidates(const std::vector<double>& query, int n);

    // functions to return top neighbors of query vector for sift data
    std::vector<std::pair<int, double>> knn_search(
    const std::vector<double>& query,
    const std::vector<std::vector<float>>& dataset,
    int n, int k);

    // functions to return top neighbors of query vector for mnist data
    std::vector<std::pair<int, double>> knn_search_mnist(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& dataset,
    int n, int k);

    //brute force for mnist data
    std::vector<std::pair<int, double>> brute_force_knn(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& dataset,
    int k);

    //brute force for sift data
    std::vector<std::pair<int, double>> brute_force_knn_sift(
    const std::vector<double>& query,                   
    const std::vector<std::vector<float>>& dataset,    
    int k);

    //range search for sift data
    std::vector<std::pair<int, double>> range_search_sift(
    const std::vector<double>& query,
    const std::vector<std::vector<float>>& dataset,
    int n,
    int radius);

    //range search for mnist data
    std::vector<std::pair<int, double>> range_search_mnist(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& dataset,
    int n,
    int radius);
    
private:
    std::mt19937_64 rng;            //random gengerator
};

#endif
