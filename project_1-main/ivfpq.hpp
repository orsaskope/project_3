#ifndef IVFPQ_HPP
#define IVFPQ_HPP

#include "parameters.hpp"
#include <iostream>
#include <random>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <chrono>
#include <cmath>

typedef vector<vector<float>> imagesVector;
typedef vector<float> floatVec;

struct IVFPQ {
    int seed;
    int kclusters;
    int nprobe;
    int n;
    int r;
    int image_size;
    int m;
    int nbits;
    bool pq_sample;

    vector<vector<floatVec>>codebooks;
    vector<vector<vector<int>>>codes;

    imagesVector centroids; // The centroid's index in this vector, is the centroid's index in inverted_lists
    vector<imagesVector> inverted_lists;

    vector<vector<pair<int, floatVec>>>idVec;
    vector<pair<int,float>>centroids_dist;
    
    IVFPQ(int seed_, int kclusters_, int nprobe_, int n_, int r_, int image_size_, int m_, int nbits_, bool pq_sample_);
};


void clustering(imagesVector&, IVFPQ*);
void trainPQ(IVFPQ*, imagesVector);
void encodePQ(IVFPQ*, imagesVector);
vector<floatVec> clusteringSubspaces(IVFPQ*, vector<floatVec>, int, int);
vector<int> QueryCentroidSearch(IVFPQ*, floatVec);
pair<vector<pair<int, float>>, vector<int>> QueryVectorSearch(IVFPQ*, floatVec, int, vector<int>, FILE*, imagesVector&);
vector<vector<float>> DistanceTables(IVFPQ*, floatVec);

vector<pair<int,float>> bruteForce(IVFPQ*, floatVec, int, FILE*, imagesVector&);
void getNewCentroid(imagesVector&, IVFPQ*, floatVec, std::default_random_engine&, int);
void updateCentroids(IVFPQ*);
floatVec findMinDistanceToCentroids(imagesVector&, IVFPQ*);
void assignToNearestCentroid(imagesVector&, IVFPQ*);
void IvfpqSearch(imagesVector& dataset, IVFPQ* ivfpq, imagesVector queryfile, string output);

// float euclideanDist(floatVec&, floatVec&, int);
// bool comparePairs(pair<int, float>, pair<int, float>);

#endif