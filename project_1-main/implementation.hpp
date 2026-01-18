#ifndef IMPLEMENTATION_HPP
#define IMPLEMENTATION_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "hypercube.hpp"          // Hypercube Class
#include "LSH.hpp"          // LSH class
#include "parameters.hpp"   // Params, MNISTData, SIFTData structs

//euclidean distance function

double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);

//these are the functions that read the sift and mnist data

MNISTData readInputMnist(FILE* fd);
SIFTData readInputSift2(FILE* fd);

// Run lsh for mnist
void run_mnist_experiment_lsh(Params* p, MNISTData& mnist);

// Run lsh for sift
void run_sift_experiment_lsh(Params* p, SIFTData& sift);

// Run hyperqube for mnist
void run_mnist_experiment_hypercube(Params* p, MNISTData& mnist);

// Run hyperqube for sift
void run_sift_experiment_hypercube(Params* p, SIFTData& sift);

// Brute force for mnist

std::vector<std::pair<int, double>> brute_force_knn(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& dataset,
    int k);

// Brute force for sift

std::vector<std::pair<int, double>> brute_force_knn_sift(
    const std::vector<double>& query,                   
    const std::vector<std::vector<float>>& dataset,    
    int k);



#endif // IMPLEMENTATION_HPP
