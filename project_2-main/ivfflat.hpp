#ifndef IVFFLAT_HPP
#define IVFFLAT_HPP

#include "parameters.hpp"
#include <iostream>
#include <random>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <chrono>

typedef vector<vector<float>> imagesVector;
typedef vector<float> floatVec;

struct IVFFLAT {
    int seed;
    int kclusters;
    int nprobe;
    int n;
    int r;
    int image_size;

    imagesVector centroids; // The centroid's index in this vector, is the corresponding cluster's index in inverted_lists
    vector<imagesVector> inverted_lists;

    vector<vector<pair<int, floatVec>>>idVec;   // For query phase keeps index and distance
    vector<pair<int,float>>centroids_dist;
    
    IVFFLAT(int seed_, int kclusters_, int nprobe_, int n_, int r_, int image_size_);
};

void IvfflatSearch(imagesVector&, IVFFLAT*, imagesVector, string output);
void IvfflatSearch_KNN(imagesVector&, IVFFLAT*, string);

void clustering(imagesVector&, IVFFLAT*);
void getNewCentroid(imagesVector&, IVFFLAT*, floatVec, std::default_random_engine&, int);
void updateCentroids(IVFFLAT*);
floatVec findMinDistanceToCentroids(imagesVector&, IVFFLAT*);
void assignToNearestCentroid(imagesVector&, IVFFLAT*);
float silhouette (imagesVector, IVFFLAT*);

vector<int> QueryCentroidSearch(IVFFLAT*, floatVec);
bool comparePairs(pair<int, float>, pair<int, float>);
pair<vector<pair<int, float>>, vector<int>> QueryVectorSearch(IVFFLAT*, floatVec, int, vector<int>, FILE*, imagesVector&);
vector<pair<int,float>> bruteForce(IVFFLAT*, floatVec, int, FILE*, imagesVector&);

float euclideanDist(floatVec& a, floatVec& b, int image_size);


#endif