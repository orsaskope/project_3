#include "hypercube.hpp"
#include "implementation.hpp"

Hypercube::Hypercube(int kproj_, int dim_, double w_, unsigned seed_, int M_, int probes_)
    : kproj(kproj_), dim(dim_), w(w_), seed(seed_), M(M_), probes(probes_), rng(seed_)
{
    std::normal_distribution<double> nd(0.0, 1.0);       // For random projection vectors
    std::uniform_real_distribution<double> ud(0.0, w);   // For random shift b
    std::uniform_int_distribution<int> dist(1, INT32_MAX);

    std::cout << "Hypercube constructor has been used\n";

    // Resize random projection vectors
    a.resize(kproj, std::vector<double>(dim));
    b.resize(kproj);
    r.resize(kproj);
    t.resize(kproj);
    hto10.resize(kproj);

    // Fill a and b with random values
    for (int i = 0; i < kproj; ++i) {
        for (int d = 0; d < dim; ++d) {
            a[i][d] = nd(rng);  // Gaussian random vector for projection
        }
        b[i] = ud(rng);        // Random shift for projection
        r[i] = dist(rng);
        t[i] = dist(rng);
    }
}

Hypercube::VertexResult Hypercube::compute_vertex(const std::vector<double>& point,int index) {
    std::string key;
    key.reserve(kproj); // reserve space so the string doesn't grow each time

    for (int i = 0; i < kproj; ++i) {
        double dot = 0.0;

        // calculate a_i · x (dot product)
        for (int d = 0; d < dim; ++d) {
            dot += a[i][d] * point[d];
        }

        // add shift and divide by w (this decides the side of the hyperplane)
        double val = (dot + b[i]) / w;

        // convert to bit depending on the sign of val 
        if (val >= 0) {
            key = key + "1";
        } else {
            key = key + "0";
        }
    }

    // return the point's index and its binary key in the hypercube
    return { index,key };
}






// insert point into hypercube
void Hypercube::insert(int index, const std::vector<double>& point) {
    VertexResult vertex = compute_vertex(point,index); // get struct
    hypercubeTable[vertex.vertex_key].push_back(vertex); // use the string key
}


//This function just prints some info about the hypercube table
void Hypercube::print_table_info() {
    std::cout << "Hypercube table has " << hypercubeTable.size() << " vertices.\n";
    for (const auto& [key, vec] : hypercubeTable) {
        std::cout << "Vertex " << key << " has " << vec.size() << " points.\n";
    }
}


std::vector<Hypercube::VertexResult> Hypercube::query(const std::vector<double>& point, int M) {
    std::vector<VertexResult> candidates;

    // Get the binary key of the query point
    VertexResult query_vertex = compute_vertex(point, -1);
    const std::string& query_key = query_vertex.vertex_key;

    std::unordered_set<std::string> visited; // keeps track of which vertices we've already checked
    visited.insert(query_key);

    // First, check the bucket query vector
    auto it = hypercubeTable.find(query_key);
    if (it != hypercubeTable.end()) {
        // Push all points in this bucket to candidates
        candidates.insert(candidates.end(), it->second.begin(), it->second.end());
    }

    std::queue<std::string> q; // queue for BFS on neighbors (flip 1 bit at a time)
    q.push(query_key);

    int visited_vertices = 1; // count how many vertices we have examined

    // Search neighboring vertices until we hit the limit (probes) or enough points (M) or the queue is empty
    while (!q.empty() && visited_vertices < probes && (int)candidates.size() < M) {
        std::string current = q.front();
        q.pop();

        // Try flipping each bit in the current key to explore neighbors
        for (int i = 0; i < (int)current.size(); ++i) {
            std::string flipped = current;
            // flip bit i (change 0→1 and 1→0)
            if (flipped[i] == '0') {
                flipped[i] = '1';
            } else {
                flipped[i] = '0';
            }

            // skip if we've already checked this vertex before
            if (!visited.insert(flipped).second)
                continue;

            // if this new vertex exists in the table, add its points
            auto it2 = hypercubeTable.find(flipped);
            if (it2 != hypercubeTable.end()) {
                //to avoid checking empty buckets we only increase visited_vertices when we know the bucket is not empty,then push the points of the bucket into candidates
                visited_vertices++;
                candidates.insert(candidates.end(), it2->second.begin(), it2->second.end());
                if ((int)candidates.size() >= M)
                    break; // stop if we reached our limit
            }

            // we push it only if we still have probes left
            if (visited_vertices < probes)
                q.push(flipped);

            // stop conditions again (probes and M)
            if (visited_vertices > probes || (int)candidates.size() >= M)
                break;
        }
    }

    // If we collected more than M candidates, keep only the first M
    if ((int)candidates.size() > M)
        candidates.resize(M);


    return candidates;
}






// Finds the N nearest neighbors from a list of candidates
std::vector<Hypercube::VertexResult> Hypercube::find_top_n_neighbors(
    const std::vector<double>& query,
    const std::vector<VertexResult>& candidates,
    int N,
    const std::vector<std::vector<double>>& dataset
) {
    std::vector<std::pair<double, Hypercube::VertexResult>> dist_vec;

    // Compute distances in a simple loop
    for (const auto& v : candidates) {
        double dist = euclidean_distance(query, dataset[v.index]);
        dist_vec.push_back({dist, v});
    }

    // Sort all distances 
    std::sort(dist_vec.begin(), dist_vec.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Keep only top N neighbors
    if ((int)dist_vec.size() > N) {
        dist_vec.resize(N);
    }

    // Extract just the VertexResults from dist_vec
    std::vector<Hypercube::VertexResult> top_n;
    for (const auto& p : dist_vec) {
        top_n.push_back(p.second);
    }

    return top_n;
}

// this is for mnist data
std::vector<Hypercube::VertexResult> Hypercube::range_search(
    const std::vector<double>& query_point,
    int radius,
    int M,
    const std::vector<std::vector<double>>& dataset
) {
    std::vector<VertexResult> results;
    std::unordered_set<int> seen;  // to avoid adding the same point twice

    // First get some candidate points from the hypercube search
    auto candidates = query(query_point, M);

    // MNIST vectors are normalized, so we scale the radius too
    double norm_radius = static_cast<double>(radius) / 255.0;

    // Check which of the candidates are actually within the distance R
    for (const auto& v : candidates) {
        // if we already checked this point before, skip it
        if (seen.find(v.index) != seen.end()) continue; // skip duplicates
        seen.insert(v.index);

        double dist = euclidean_distance(query_point, dataset[v.index]);
        if (dist <= norm_radius) {
            results.push_back(v);
        }
    }



    return results;
}


std::vector<Hypercube::VertexResult> Hypercube::range_search_sift(
    const std::vector<double>& query_point,
    int radius,
    int M,
    const std::vector<std::vector<double>>& dataset
) {
    std::vector<VertexResult> results;
    std::unordered_set<int> seen;  // same idea, avoid duplicates

    // Get candidate points from the hypercube
    auto candidates = query(query_point, M);

    // No normalization here because sift data stays raw
    for (const auto& v : candidates) {
        // if we already checked this point before, skip it
        if (seen.find(v.index) != seen.end()) continue;
        seen.insert(v.index);

        double dist = euclidean_distance(query_point, dataset[v.index]);
        // we use the raw R since sift data is also raw
        if (dist <= radius) {
            results.push_back(v);
        }
    }



    return results;
}



