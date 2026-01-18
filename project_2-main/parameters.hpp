#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>
#include <cstdio>
#include <vector>
using namespace std;

struct Params{
    string input;    // Input file
    string query;    // Query
    string o;        // Output file
    int k;          // Number of LSH functions for g
    int l;          // Number of hashtables
    double w;       // Cell size on the straight line
    int n;          // Number of nearest
    int r;          // Search radius
    string type;    // sift/mnist
    bool range;     // If false->No aerea search

    int kproj;      // Projection points
    int m;          // Max points to check(hypercube) OR number of subvectors(ivfpq)
    int probes;     // Max cube vertices

    int kclusters;  // Number of clusters
    int nprobe;     // Number of clusters to check
    int seed;       // For rand
    int nbits;      // 2^nbits subspaces
    int algorithm;  // lsh = 0, hypercube = 1, ivfflat = 2, ivfpq = 3

    bool pq_sample;

    Params();
};


struct MNISTData {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    int image_size = 0; 
    std::vector<std::vector<unsigned char>> images;
};

struct SIFTData {
    int number_of_vectors = 0;     
    int v_dim = 320;     
    std::vector<std::vector<float>> vectors; 
};

void validArgument(char* argument, int argc, int i);
Params* ArgsParser(int argc, char* argv[]);
void initializeParams(Params* params);
string returnType(Params* p);
void printParameters(Params* p);

#endif