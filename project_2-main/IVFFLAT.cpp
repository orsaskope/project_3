#include "ivfflat.hpp"


using namespace std;

IVFFLAT::IVFFLAT(int seed_, int kclusters_, int nprobe_, int n_, int r_, int image_size_)
    :seed(seed_), kclusters(kclusters_), nprobe(nprobe_), n(n_), r(r_), image_size(image_size_) {
    centroids.reserve(kclusters);
    cout << "IVFFLAT constructor used" << endl;
}

void IvfflatSearch_KNN(imagesVector& dataset, IVFFLAT* ivfflat, string output) {
    // Training
    clustering(dataset, ivfflat);

    FILE* fout = fopen(output.c_str(), "w");
    if (!fout) {
        perror("Failed to open output file");
        exit(errno);
    }

    // ANN for all dataset
    for (int i = 0; i < (int)dataset.size(); i++) {
        floatVec& q = dataset[i];

        vector<int> top_clusters = QueryCentroidSearch(ivfflat, q);
        auto results = QueryVectorSearch(ivfflat, q, i, top_clusters, nullptr, dataset);

        vector<pair<int,float>>& nn = results.first;

        // Output
        fprintf(fout, "NODE %d:", i);
        for (auto& p : nn) {
            fprintf(fout, " %d", p.first);
        }
        fprintf(fout, "\n");
    }

    fclose(fout);
}

void IvfflatSearch(imagesVector& dataset, IVFFLAT* ivfflat, imagesVector queryfile, string output) {
    // using namespace std::chrono;
    // auto start = high_resolution_clock::now();

    // FILE* fout = fopen(output.c_str(), "w");
    // if (!fout) {
    //     perror("Failed to open output file");
    //     exit(errno);
    // }
    // fprintf(fout, "IVFFlat\n");
    // fflush(fout);

    // clustering(dataset, ivfflat);
    // auto end = high_resolution_clock::now();
    // duration<double> elapsed = end - start;
    // cout << "\nTraining completed in " << elapsed.count() << " seconds." << endl;
    // //float silhouette_score = silhouette (dataset, ivfflat);
   
    // double total_af = 0.0;           // Sum of all Approximation Factors (distanceApproximate / distanceTrue)
    // double total_recall = 0.0;       // Sum of all the true nearest neighbours that ivfflat found/N
    // double total_ivfflat_time = 0.0;   // Total IVFFLAT time
    // double total_brf_time = 0.0;     // Total brute force time
    // size_t Q = queryfile.size();       // Queries searched
    // int N = ivfflat->n;             // Nearest neighbours
    // cout << "searching.." << endl;
    
    // for (size_t i = 0; i < queryfile.size(); i++) {
    //     fprintf(fout, "\nQuery: %zu\n", i);
    //     fflush(fout);
        
    //     // Search for queries' approximate nearest neighbours, count time.
    //     auto t1 = high_resolution_clock::now();
    //     vector<int> top_clusters = QueryCentroidSearch(ivfflat, queryfile[i]);
    //     pair<vector<pair<int, float>>, vector<int>> results = QueryVectorSearch(ivfflat, queryfile[i], i, top_clusters, fout, dataset);
    //     vector<pair<int, float>> nn_res = results.first;
    //     vector<int> range_res = results.second;
    //     auto t2 = high_resolution_clock::now();
    //     double curr_ann_time = duration<double>(t2 - t1).count();
    //     total_ivfflat_time += curr_ann_time;   // Add the current time to the total

    //     // Search for queries actual nearest neighbours, cout time.
    //     auto t3 = high_resolution_clock::now();
    //     vector<pair<int, float>> brute_res = bruteForce(ivfflat, queryfile[i], i, fout, dataset);
    //     auto t4 = high_resolution_clock::now();
    //     double curr_true_time = duration<double>(t4 - t3).count();
    //     total_brf_time += curr_true_time;

    //     double curr_af = nn_res[0].second / brute_res[0].second;
    //     total_af += curr_af;

    //     // --- Recall@N ---
    //     double recall = 0.0;
    //     int correct = 0;
    //     for (int ivfflat_n = 0; ivfflat_n < N; ivfflat_n++) {
    //         int ivfflat_idx = nn_res[ivfflat_n].first;
    //         for (int brute_n = 0; brute_n < N; brute_n++) {
    //             if (brute_res[brute_n].first == ivfflat_idx) {
    //                 correct++;
    //                 break;
    //             }
    //         }
    //     }
    //     recall = (double)correct / N;
    //     total_recall += recall;

    //     for (int j = 0; j < ivfflat->n; ++j) {
    //         fprintf(fout, "Nearest neighbor-%d: %d\n", j + 1, nn_res[j].first);
    //         fprintf(fout, "distanceApproximate: %.6f\n", nn_res[j].second);
    //         fprintf(fout, "distanceTrue: %.6f\n", brute_res[j].second);
    //         fflush(fout);
    //     // fprintf(fout, "distanceTrue: %.6f\n", res[j].second);
    //     }
    //     if (ivfflat->r) {
    //             fprintf(fout, "R-near neighbors:\n");
    //             for (size_t i = 0; i < range_res.size(); i++)
    //                 fprintf(fout, "%d\n",range_res[i]);
    //     }

    //     // vector<int> top_clusters = QueryCentroidSearch(ivfflat, queryfile[i]);
    //     // cout << "query " << i << " top clusters: ";
    //     // for (int c : top_clusters) cout << c << " ";
    //     // cout << endl;
    //     // vector<pair<int, float>> res = QueryVectorSearch(ivfflat, queryfile[i], top_clusters, fout, dataset);

    //     // for (int j = 0; j < ivfflat->n; j++) {
    //     //    cout << "index: " << res[j].first << " distance: " << res[j].second << endl;
    //     // }
    //     // cout << "query " << i << " end" << endl;
    // }
    // fprintf(fout, "Average AF: %.6f\n", total_af / Q);
    // fprintf(fout, "Recall@N: %.6f\n", total_recall / Q);
    // fprintf(fout, "QPS: %.6f\n", Q / total_ivfflat_time);
    // fprintf(fout, "tApproximateAverage: %.6f\n", total_ivfflat_time / Q);
    // fprintf(fout, "tTrueAverage: %.6f\n", total_brf_time / Q);
    // //fprintf(fout, "silhouette score: %.6f\n", silhouette_score);

    // fclose(fout);
    // return;
}

/*-------------------------------------CLUSTERING---------------------------------------*/

void clustering(imagesVector& dataset, IVFFLAT* ivfflat) {
    int seed = ivfflat->seed;
    int kclusters = ivfflat->kclusters;

    // initializing centroids by chosing "kclusters" random images. 
    // Generate random num for index.
    default_random_engine generator(seed);
    uniform_int_distribution<int> dist(0, dataset.size() - 1);

    imagesVector sample;

    int sample_sz = sqrt(dataset.size());
    sample.reserve(sample_sz);

    vector<int> sample_idx;
    while((int)sample.size() < sample_sz) {
        int idx = dist(generator);
        if (find(sample_idx.begin(), sample_idx.end(), idx) == sample_idx.end()) {
            sample.push_back(dataset[idx]);
            sample_idx.push_back(idx);
        }
    }

    ivfflat->inverted_lists.resize(ivfflat->kclusters);
    ivfflat->idVec.resize(ivfflat->kclusters);

    int idx = dist(generator) % sample.size();
    ivfflat->centroids.push_back(sample[idx]);
    ivfflat->inverted_lists[0].push_back(sample[idx]);


    /*For the initialization of the first kclusters centroids:
    The vector that is furthest from its closest centroid has better chances to
    become a centroid. When we choose the centroids, assign the vectors to the
    corresponding clusters */
    for (int i = 1; i < kclusters; i++) {
        cout << "Training..." << endl;
        floatVec D = findMinDistanceToCentroids(sample, ivfflat);
        getNewCentroid(sample, ivfflat, D, generator, i);
    }

    // After the initialization, assign vectors to their nearest centroid and
    // recalculate centroids. Repeat until there are not many changes.
    assignToNearestCentroid(dataset, ivfflat);
    for (int i = 0; i < 15; i++) {
        imagesVector old_centroids = ivfflat->centroids;
        float change = 0.0f;

        updateCentroids(ivfflat);
        assignToNearestCentroid(dataset, ivfflat);
        
        for (int i = 0; i < ivfflat->kclusters; i++)
        change += euclideanDist(old_centroids[i], ivfflat->centroids[i], ivfflat->image_size);
        float avg_movement = change / ivfflat->kclusters;
        cout << "Iteration " << i << " | avg_movement=" << avg_movement << endl;
        if (avg_movement < 20) break;
    }
    
    return;
}

// Iterate through all the vectors and compute their distance to their closest centroid. This will help compute the probability
// of every vector becoming a centroid. We keep the distances squared for the probability function P(Xi) = D(i)^2/ΣD(j)
floatVec findMinDistanceToCentroids(imagesVector& dataset, IVFFLAT* ivfflat) {
    floatVec D; // Will hold the distance of every vector to its nearest centroid (squared).
    for (size_t i = 0; i < dataset.size(); i++) {
        // If current vector is a centroid, update D with 0 (so we don't lose indexes).
        if (std::find(ivfflat->centroids.begin(), ivfflat->centroids.end(), dataset[i]) != ivfflat->centroids.end()) {
            D.push_back(0);
            continue;
        }

        // Vector is not a centroid
        float dist = std::numeric_limits<float>::max();
        for (size_t j = 0; j < ivfflat->centroids.size(); j++) {
            float curr_dist = euclideanDist(dataset[i], ivfflat->centroids[j], ivfflat->image_size);
            if (curr_dist < dist) dist = curr_dist;
        }
        D.push_back(dist * dist);
    }
    return D;
}

void getNewCentroid(imagesVector& dataset, IVFFLAT* ivfflat, floatVec D, std::default_random_engine& generator, int idx) {
    floatVec p_sums;    // Wil hold the distance to closest vector of ALL the previous vectors and the current one.
    float dist_sum = 0.0f;
    for (size_t i = 0; i < D.size(); i++) {
        dist_sum += D[i];
        p_sums.push_back(dist_sum);
    }
    
    float total_sum = p_sums.back();    // Total sum of all vectors' distances, for random index initialization.
    for (size_t i = 0; i < D.size(); i++) {
        p_sums[i] /= total_sum;
    }

    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float x = dis(generator);

    auto iter = lower_bound(p_sums.begin(), p_sums.end(), x);
    int r = iter - p_sums.begin();

    ivfflat->centroids.push_back(dataset[r]);
    ivfflat->inverted_lists[idx].push_back(dataset[r]);
    cout << "got new centroid index in centroids: " << idx;
}

// Assigns every vector to its closest centroid.
void assignToNearestCentroid(imagesVector& dataset, IVFFLAT* ivfflat) {
    ivfflat->inverted_lists.clear();
    ivfflat->inverted_lists.resize(ivfflat->kclusters);
    
    ivfflat->idVec.clear();
    ivfflat->idVec.resize(ivfflat->kclusters);
    
    // iterate through every vector and find the closest centroid. Assign vector to the corresponding cluster.
    for (size_t i = 0; i < dataset.size(); i++) {
        // Initialize dist and nearest centroid so that we can find minimum distance and assign current vector to closest centroid's cluster.
        float dist = std::numeric_limits<float>::max(); 
        int nearest = 0; 
        floatVec a = dataset[i];

        //if (std::find(ivfflat->centroids.begin(), ivfflat->centroids.end(), a) != ivfflat->centroids.end()) continue;
        
        for (size_t j = 0; j < ivfflat->centroids.size(); j++) {
            floatVec b = ivfflat->centroids[j];  // got current centroid
            float temp_dist = euclideanDist(a, b, ivfflat->image_size);
            if (temp_dist < dist) {
                dist = temp_dist;
                nearest = j;
            }
        }
        ivfflat->inverted_lists[nearest].push_back(a);
        ivfflat->idVec[nearest].push_back({i, a});
    }  
    cout << "assigned everything in clusters" << endl;  
}

void updateCentroids(IVFFLAT* ivfflat) {
    for (int i = 0; i < ivfflat->kclusters; i++) {
        imagesVector curr_cl = ivfflat->inverted_lists[i]; // Got current cluster
        if (curr_cl.empty())    continue;
        
        // This vector will hold the average of the current clusters' vectors coordinates
        floatVec new_cntr;
        new_cntr.resize(ivfflat->image_size, 0.0f);

        for (size_t j = 0; j < curr_cl.size(); j++) {
            for (size_t k = 0; k < curr_cl[j].size(); k++) {
                new_cntr[k] += curr_cl[j][k];
            }
        }
        for (int k = 0; k < ivfflat->image_size; k++) {
            new_cntr[k] /= curr_cl.size();
        }
        ivfflat->centroids[i] = new_cntr;
    }
    
}

/*-------------------------------------QUERY SEARCH PHASE--------------------------------------*/

// Search the nprobe closest centroids and return a vector of the centroid's index and the distance.
vector<int> QueryCentroidSearch(IVFFLAT* ivfflat, floatVec q) {
    vector<pair<int,float>>centroids_dist;
    
    for (int i = 0; i < ivfflat->kclusters; i++) {
        float d = euclideanDist(q, ivfflat->centroids[i], ivfflat->image_size);
        centroids_dist.push_back({i, d});
    }
    sort(centroids_dist.begin(), centroids_dist.end(), comparePairs);   // Sorting to find the closest

    vector<int> top_clusters;
    for (int j = 0; j < ivfflat->nprobe; j++)
        top_clusters.push_back(centroids_dist[j].first);    // Get the nprobe closest
    return top_clusters;
}

// Search in the nprobe closest clusters, the nearest neighbours and sort them
pair<vector<pair<int, float>>, vector<int>> QueryVectorSearch(IVFFLAT* ivfflat, floatVec q, int q_idx, vector<int> top_clusters, FILE* fout, imagesVector& dataset) {
    vector<pair<int, float>> res;
    vector<int> range_res;

    for (size_t i = 0; i < top_clusters.size(); i++) {
        int curr_cl = top_clusters[i];

        for (size_t j = 0; j < ivfflat->idVec[curr_cl].size(); j++) {
            int idx = ivfflat->idVec[curr_cl][j].first;       // index of vector
            
            if (idx == q_idx)   continue;
            floatVec x = ivfflat->idVec[curr_cl][j].second;   // vector
            float dist = euclideanDist(q, x, ivfflat->image_size);
            res.push_back({idx, dist});
        }
    }
    sort(res.begin(), res.end(), comparePairs);
    
    if (ivfflat->r) {
        for (size_t i = 0; i < res.size(); i++) {
            if (res[i].second < ivfflat->r)
            range_res.push_back(res[i].first);
        }
    }
    
    int N = ivfflat->n;
    if ((int)res.size() < N) N = res.size();
    res.resize(N);

    return {res, range_res};
}


/*----------------------------------HELPER FUNCTIONS--------------------------------------------*/

// tbd: ?xreiazetai sqrt? afoy me noiazei h sygkrish apostasewn k dn me noiazei h akrivhs apostash
float euclideanDist(floatVec& a, floatVec& b, int image_size) {
    float dist = 0.0;
    for (int i = 0; i < image_size; i++) 
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(dist);
}

float silhouette(imagesVector dataset, IVFFLAT* ivfflat) {
    // cout << "\nCentroid-Cluster Consistency " << endl;
    // for (int i = 0; i < ivfflat->kclusters; i++) {
    //     if (ivfflat->inverted_lists[i].empty()) continue;

    //     float avg_to_self = 0.0f;
    //     for (auto &v : ivfflat->inverted_lists[i])
    //         avg_to_self += euclideanDist(v, ivfflat->centroids[i], ivfflat->image_size);
    //     avg_to_self /= ivfflat->inverted_lists[i].size();

    //     // Μετρά και απόσταση προς τα διπλανά centroids για να δεις αν επικαλύπτονται
    //     float min_other = std::numeric_limits<float>::max();
    //     for (int j = 0; j < ivfflat->kclusters; j++) {
    //         if (j == i) continue;
    //         float d = euclideanDist(ivfflat->centroids[i], ivfflat->centroids[j], ivfflat->image_size);
    //         if (d < min_other) min_other = d;
    //     }

    //     cout << "Cluster " << i
    //          << " | size=" << ivfflat->inverted_lists[i].size()
    //          << " | avg_dist_to_own_centroid=" << avg_to_self
    //          << " | min_dist_to_nearest_other_centroid=" << min_other
    //          << endl;
    // }

    float score = 0.0f;
    for (int i = 0; i < ivfflat->kclusters; i++) {
        imagesVector cluster = ivfflat->inverted_lists[i];
        //floatVec centroid = ivfflat->centroids[i];
        for (size_t j = 0; j < cluster.size(); j++) {
            floatVec vec = cluster[j];
            float avg_dist_curr = 0.0f;
            for (size_t neigh = 0; neigh < cluster.size(); neigh++) {
                avg_dist_curr += euclideanDist(vec, cluster[neigh], ivfflat->image_size);
            }
            avg_dist_curr /= cluster.size();

            int indx_next_centroid = 0;
            float closest_dist = std::numeric_limits<float>::max();
            for (size_t k = 0; k < ivfflat->centroids.size(); k++) {
                if ((int)k == i) continue;
                int dist = euclideanDist(vec, ivfflat->centroids[k], ivfflat->image_size);
                if (dist < closest_dist) {
                    closest_dist = dist;
                    indx_next_centroid = k;
                }
            }

            float avg_dist_next = 0.0f;
            imagesVector next_cluster = ivfflat->inverted_lists[indx_next_centroid];
            for (size_t neigh = 0; neigh < next_cluster.size(); neigh++) {
                avg_dist_next += euclideanDist(vec, next_cluster[neigh], ivfflat->image_size);
            }
            avg_dist_next /= (int)next_cluster.size();

            if (avg_dist_curr > avg_dist_next)
                cout << "silhouete score: " << (avg_dist_next - avg_dist_curr) / avg_dist_curr << endl; 
            else
                cout << "ssilhouete score: " << (avg_dist_next - avg_dist_curr) / avg_dist_next << endl;
            
            score += (avg_dist_curr > avg_dist_next) ? (avg_dist_next - avg_dist_curr) / avg_dist_curr : (avg_dist_next - avg_dist_curr) / avg_dist_next;
        }
        score /= (int)dataset.size();
    }
    return score;
}



bool comparePairs(pair<int, float> a, pair<int, float> b) {
    return a.second < b.second;
}

vector<pair<int,float>> bruteForce(IVFFLAT* ivfflat, floatVec q, int q_idx, FILE* fout, imagesVector& dataset) {
    vector<pair<int, float>> brute_res;

    for (size_t i = 0; i < dataset.size(); i++) {
        if (q_idx == (int)i) continue;
        float d = euclideanDist(q, dataset[i], ivfflat->image_size);
        brute_res.push_back({(int)i, d});
    }

    sort(brute_res.begin(), brute_res.end(), comparePairs);
    if ((int)brute_res.size() > ivfflat->n)
        brute_res.resize(ivfflat->n);
    return brute_res;
}
