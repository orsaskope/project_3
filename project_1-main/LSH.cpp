#include "LSH.hpp"
#include "implementation.hpp"

LSH::LSH(int L_, int k_, int dim_, double w_, unsigned seed_)
    : L(L_), k(k_), dim(dim_), w(w_), seed(seed_), rng(seed_)
{
    // random distributions: normal for 'a', uniform [0,w) for 'b'
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, w);

    std::cout << "LSH constructor has been used \n";

    // allocate L tables, each has k projections of length dim
    a.resize(L, std::vector<std::vector<double>>(k, std::vector<double>(dim)));
    b.resize(L, std::vector<double>(k, 0.0));
    hashTables.resize(L); // one hash table per L

    // fill random projection vectors 'a' and shifts 'b'
    for (int l = 0; l < L; ++l) {
        for (int i = 0; i < k; ++i) {
            for (int d = 0; d < dim; ++d) {
                a[l][i][d] = nd(rng); // using the gaussian randoms
            }
            b[l][i] = ud(rng);        // random shift in [0, w)
        }
    }

    // random numbers to use for combining hashes 
    r.resize(L, std::vector<long long>(k));
    std::uniform_int_distribution<long long> ri_dist(1, 1e9);

    for (int l = 0; l < L; ++l) {
        for (int i = 0; i < k; ++i) {
            r[l][i] = ri_dist(rng); // different weight per hash
        }
    }
}



std::vector<long long> LSH::compute_h(const std::vector<double>& point, int tableIdx) {
    // this will hold the k hash values (one for each projection)
    std::vector<long long> hvals(k);

    for (int i = 0; i < k; ++i) {
        double dot = 0.0;

        // compute dot product between a and point
        for (int d = 0; d < dim; ++d)
            dot += a[tableIdx][i][d] * point[d];

        // use b and divide by w 
        double val = (dot + b[tableIdx][i]) / w;

        // take floor of val
        hvals[i] = static_cast<long long>(std::floor(val));
    }

    return hvals;
}


LSH::GResult LSH::compute_g(const std::vector<double>& point, int tableIdx, int n ,int index) {
    static const unsigned long long M = 4294967291ULL;
    auto hvals = compute_h(point, tableIdx);

    // (a+b) mod M =((a mod M)+(b mod M)) mod M therefore
    // sum=(r1​h1​+r2​h2​+…+rk​hk​) mod M is equivalent to sum=(((r1​h1​) mod M)+ ((r2​h2​) mod M)+ ..... + ((rk​hk​) mod M)) mod M which is better because 
    // that way we get to also search for the case of overflow.
    // Doing sum %= M at every step: 
    // 1)Keeps the intermediate sum from overflowing
    // 2)Preserves the correct modular arithmetic result

    unsigned long long sum = 0;
    for (int i = 0; i < k; ++i) {
        long long h = hvals[i];

        // compute h mod M safely (handles negative h)
        // uses hmod =( (h mod M) + M ) mod M to normalize negatives in [0,M-1]
        long long hmod = (h % static_cast<long long>(M) + static_cast<long long>(M)) % static_cast<long long>(M);

        // cast to unsigned long long for modular arithmetic so we avoid possible arithmetic problems
        unsigned long long hnorm = static_cast<unsigned long long>(hmod);

        sum += (r[tableIdx][i] * hnorm) % M;
        sum %= M;
    }

    unsigned long long TableSize = n / 4;  
    unsigned long long ID = sum % M;       // the locality-sensitive ID
    unsigned long long g_val = ID % TableSize;

    return { index, std::to_string(g_val), ID };
}

//insert function that puushes vectors into hashtables
void LSH::insert(int index, const std::vector<double>& point, int n) {
    for (int l = 0; l < L; ++l) {
        GResult g_res = compute_g(point, l, n , index);
        hashTables[l][g_res.g_key].push_back(g_res);

    }
}


//simple function to print info about the hashtables
void LSH::print_tables() {
    for (int l = 0; l < L; ++l) {
        std::cout << "Table " << l << " has " << hashTables[l].size() << " buckets\n";
    }
}



std::vector<LSH::GResult> LSH::get_candidates(const std::vector<double>& query, int n) {
    std::vector<GResult> candidates;           
    std::unordered_set<int> seen;              

    for (int l = 0; l < L; ++l) {              // check each of the L hash tables
        // compute the g hash key for this query in table l (index = -1 because we dont care about it at this point)
        GResult query_g = compute_g(query, l, n, -1);

        // look for this g-key 
        auto it = hashTables[l].find(query_g.g_key);
        if (it == hashTables[l].end()) continue; // if bucket doesn't exist continue

        // go through every vector stored in this bucket
        for (const auto& entry : it->second) {

            // avoid duplicates (same vector can appear in multiple tables)
            if (!seen.count(entry.index)) {
                seen.insert(entry.index);
                candidates.push_back(entry);   // add this vector to candidate list
            }
        }
    }

    return candidates; 
}


std::vector<std::pair<int, double>> LSH::knn_search(
    const std::vector<double>& query,
    const std::vector<std::vector<float>>& dataset,
    int n, int k)
{
    // Get candidate points using LSH hashing
    std::vector<GResult> candidates = get_candidates(query, n);

    // Compute distances for each candidate
    std::vector<std::pair<int, double>> distances;
    distances.reserve(candidates.size());  // reserve space

    for (const auto& c : candidates) {
        const auto& data_vec = dataset[c.index];  // get the actual vector
        std::vector<double> data_double(data_vec.begin(), data_vec.end());
        double dist = euclidean_distance(query, data_double);
        distances.emplace_back(c.index, dist); //store the index and distance
    }

    // Sort by distance
    std::sort(distances.begin(), distances.end(),
              [](auto& a, auto& b) { return a.second < b.second; });

    // Keep top-k neighbors
    if (distances.size() > static_cast<size_t>(k))
        distances.resize(k);

    return distances;
}

std::vector<std::pair<int, double>> LSH::knn_search_mnist(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& dataset,
    int n, int k)
{
    // First, get all candidate points that hashed to the same buckets
    std::vector<GResult> candidates = get_candidates(query, n);

    std::vector<std::pair<int, double>> distances;
    distances.reserve(candidates.size());  // reserve space 

    // For each candidate, compute the Euclidean distance to the query
    for (const auto& c : candidates) {
        const std::vector<double>& data_double = dataset[c.index];
        double dist = euclidean_distance(query, data_double);
        distances.emplace_back(c.index, dist);  //store the index and distance
    }

    // sort the results
    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // keep only the top k closest neighbors
    if (distances.size() > static_cast<size_t>(k))
        distances.resize(k);

    return distances;  // return the k nearest neighbors found by LSH
}






std::vector<std::pair<int, double>> LSH::range_search_mnist(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& dataset,
    int n,
    int radius)
{
    // Get candidates
    std::vector<GResult> candidates = get_candidates(query, n);

    // Radius is normalized because minst data are normalized
    double norm_radius = radius / 255.0;

    std::vector<std::pair<int, double>> in_range;
    in_range.reserve(candidates.size()); // reserve space 

    // Check which candidates fall inside the radius distance
    for (const auto& c : candidates) {
        const std::vector<double>& data_double = dataset[c.index];
        double dist = euclidean_distance(query, data_double);

        if (dist <= norm_radius) {
            in_range.emplace_back(c.index, dist); // keep (index, distance)
        }
    }

    return in_range;
}


std::vector<std::pair<int, double>> LSH::range_search_sift(
    const std::vector<double>& query,
    const std::vector<std::vector<float>>& dataset,
    int n,
    int radius)
{
    // Get candidates
    std::vector<GResult> candidates = get_candidates(query, n);

    std::vector<std::pair<int, double>> in_range;
    in_range.reserve(candidates.size()); // reserve space 

    // Check distances for the candidates
    for (const auto& c : candidates) {

        // Convert float to double 
        const std::vector<float>& data_float = dataset[c.index];
        std::vector<double> data_double(data_float.begin(), data_float.end());

        double dist = euclidean_distance(query, data_double);

        // For SIFT we do not normalize radius (raw data)
        if (dist <= radius) {
            in_range.emplace_back(c.index, dist);
        }
    }

    return in_range;
}







