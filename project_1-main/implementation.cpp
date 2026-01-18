#include "implementation.hpp"
#include <chrono>
#include <unordered_set>

double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Brute-force k-NN for MNIST
std::vector<std::pair<int, double>> brute_force_knn(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& dataset,
    int k)
{
    std::vector<std::pair<int, double>> distances;
    distances.reserve(dataset.size()); //save memory for dataset.size() elements.
    // check the euclidean distance of query vector qith every vector of the dataset
    for (size_t i = 0; i < dataset.size(); ++i) {
        double dist = euclidean_distance(query, dataset[i]);
        distances.emplace_back(i, dist); // store index and distance
    }
    // sort the distances
    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    // keep only the top k closest
    if (distances.size() > static_cast<size_t>(k))
        distances.resize(k);

    return distances;
}

std::vector<std::pair<int, double>> brute_force_knn_sift(
    const std::vector<double>& query,                   
    const std::vector<std::vector<float>>& dataset,    
    int k)
{
    std::vector<std::pair<int, double>> distances;
    distances.reserve(dataset.size()); //save memory for dataset.size() elements.
    // check the euclidean distance of query vector qith every vector of the dataset

    for (size_t i = 0; i < dataset.size(); ++i) {
        // convert float to double for distance calculation cause sift dataset is float
        std::vector<double> vec_double(dataset[i].begin(), dataset[i].end());
        double dist = euclidean_distance(query, vec_double);
        distances.emplace_back(i, dist); // store index and distance
    }
    // sort the distances
    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    // keep only the top k closest
    if (distances.size() > static_cast<size_t>(k))
        distances.resize(k);

    return distances;
}

// ===========================================================================================================================
// 🔹 LSH IMPLEMENTATION
// ===========================================================================================================================

// =========================================
// 🔹 MNIST Experiment Function
// =========================================
void run_mnist_experiment_lsh(Params* p, MNISTData& mnist) {
    std::cout << "MNIST dataset loaded with " << mnist.number_of_images << " images.\n";

    // Normalize MNIST dataset to [0, 1] by dividing by 255
    std::vector<std::vector<double>> mnist_dataset(mnist.number_of_images, std::vector<double>(mnist.image_size));
    for (int i = 0; i < mnist.number_of_images; ++i)
        for (int j = 0; j < mnist.image_size; ++j)
            mnist_dataset[i][j] = static_cast<double>(mnist.images[i][j]) / 255.0;  // Normalizing each pixel

    // 🔹 Create LSH object dynamically
    LSH* lsh = nullptr;
    lsh = new LSH(p->l, p->k, mnist.image_size, p->w, p->seed);

    for (int i = 0; i < mnist.number_of_images; ++i)
        lsh->insert(i, mnist_dataset[i], mnist.number_of_images);   //insert all the vectors of mnist dataset into lsh

    // lsh->print_tables(); //optional if you want to see info about the lsh buckets

    // --- Load queries ---
    FILE* fq = fopen("query.dat", "r");
    if (!fq) { perror("Failed to open query.dat"); exit(errno); }
    MNISTData mnist_queries = readInputMnist(fq);
    fclose(fq);
    std::cout << "Loaded " << mnist_queries.number_of_images << " query vectors.\n";

    // Open output file
    FILE* outfile = fopen(p->o.c_str(), "w");
    if (!outfile) {
        perror("Failed to open lsh_results.txt");
        exit(errno);
    }

    fprintf(outfile, "LSH\n");

    double total_AF = 0, total_recall = 0, total_t_lsh = 0, total_t_true = 0;  //variables for counting averages

    for (int qi = 0; qi < mnist_queries.number_of_images; ++qi) {
        std::vector<double> query_vec(mnist.image_size);
        for (int j = 0; j < mnist.image_size; ++j)
            query_vec[j] = static_cast<double>(mnist_queries.images[qi][j]) / 255.0;  // Normalizing every mnist query image

        fprintf(outfile, "Query: %d\n", qi);
        //counting how long a mnist vector takes into lsh 
        auto t1 = std::chrono::high_resolution_clock::now();
        auto approx_results = lsh->knn_search_mnist(query_vec, mnist_dataset, mnist.number_of_images, p->n);  //function to return n neighbors of query vector in lsh
        auto t2 = std::chrono::high_resolution_clock::now();
        double t_lsh = std::chrono::duration<double>(t2 - t1).count();
        total_t_lsh += t_lsh;
        //counting how long a mnist vector takes into brute force
        auto t3 = std::chrono::high_resolution_clock::now();
        auto true_results = brute_force_knn(query_vec, mnist_dataset, p->n);  //function to return n neighbors of query vector in brute force
        auto t4 = std::chrono::high_resolution_clock::now();
        double t_true = std::chrono::duration<double>(t4 - t3).count();
        total_t_true += t_true;

        int hits = 0;
        double sum_AF = 0;

        for (int ni = 0; ni < p->n; ++ni) {
            int idx_approx = approx_results[ni].first;      //getting the results for lsh
            double dist_approx = approx_results[ni].second;
            int idx_true = true_results[ni].first;          //getting the results for brute force
            double dist_true = true_results[ni].second;

            fprintf(outfile, "Nearest neighbor-%d: %d\n", ni + 1, idx_approx);
            fprintf(outfile, "Nearest neighbor in brute-%d: %d\n", ni + 1, idx_true);
            fprintf(outfile, "distanceApproximate: %.6f\n", dist_approx);
            fprintf(outfile, "distanceTrue: %.6f\n", dist_true);

            if (dist_true != 0) {
                sum_AF += dist_approx / dist_true;      //special case where query vector was a part of the dataset
            } else {
                sum_AF += 1;
            }

            for (int k = 0; k < p->n; ++k)
                if (approx_results[ni].first == true_results[k].first)
                    hits++;                                                 //seeing how many neighbors lsh found correctly compared to brute force
        }
        // this part is where the range search happens if it has to
        if (p->range) {
            auto range_results = lsh->range_search_mnist(query_vec, mnist_dataset, mnist.number_of_images, p->r);
            fprintf(outfile, "R-near neighbors:\n");
            for (auto& [idx, dist] : range_results)
                fprintf(outfile, "Image_ID: %d\n", idx);
        }

        double AF = sum_AF / p->n;
        double recall = static_cast<double>(hits) / p->n;
        total_AF += AF;
        total_recall += recall;

        fprintf(outfile, "Average AF: %.6f\n", AF);
        fprintf(outfile, "Recall@N: %.6f\n", recall);               //printing the averages for every query vector (optional)
        fprintf(outfile, "QPS: %.6f\n", 1.0 / t_lsh);
        fprintf(outfile, "tApproximateAverage: %.6f\n", t_lsh);
        fprintf(outfile, "tTrueAverage: %.6f\n", t_true);
        fprintf(outfile, "\n");

        
        std::cout << "Query " << qi + 1 << " done \n"; 


        // if (qi == 99) {
        //     break;          //this keeps the query vectors to 100 so we can use the dataset but not run the entire query dataset
        // }
    }

    // if you want to run the entire query dataset delete if statement in lines 172-174 and line 180

    int nq = mnist_queries.number_of_images;
    // nq = 100; 
    fprintf(outfile, "\n=== Global averages ===\n");
    fprintf(outfile, "Average AF: %.6f\n", total_AF / nq);                  //printing the averages (mandatory)
    fprintf(outfile, "Average Recall@N: %.6f\n", total_recall / nq);
    fprintf(outfile, "Average QPS: %.6f\n", nq / total_t_lsh);
    fprintf(outfile, "Average tApproximate: %.6f\n", total_t_lsh / nq);
    fprintf(outfile, "Average tTrue: %.6f\n", total_t_true / nq);

    fclose(outfile);  // Close the output file

    delete lsh;
}




// =========================================
// 🔹 SIFT Experiment Function
// =========================================
void run_sift_experiment_lsh(Params* p, SIFTData& sift) {
    std::cout << "Number of SIFT vectors: " << sift.number_of_vectors << "\n";
    std::cout << "Vector dimension: " << sift.v_dim << "\n";

    // 🔹 Create LSH dynamically
    LSH* lsh = nullptr;
    lsh = new LSH(p->l, p->k, sift.v_dim, p->w, p->seed);

    for (int i = 0; i < sift.number_of_vectors; ++i) {
        std::vector<double> vec(sift.v_dim);
        for (int j = 0; j < sift.v_dim; ++j)
            vec[j] = static_cast<double>(sift.vectors[i][j]);       //sift data is raw so we dont normalize
        lsh->insert(i, vec, sift.number_of_vectors);             //insert all the vectors of sift dataset into lsh
    }

    // lsh->print_tables(); //optional if you want to see info about the lsh buckets

    // --- Load queries ---
    FILE* fq = fopen("query.dat", "r");
    if (!fq) { perror("Failed to open query.dat"); exit(errno); }
    SIFTData sift_queries = readInputSift2(fq);
    fclose(fq);
    std::cout << "Loaded " << sift_queries.number_of_vectors << " query vectors.\n";

    // Open output file
    FILE* outfile = fopen(p->o.c_str(), "w");
    if (!outfile) {
        perror("Failed to open lsh_results.txt");
        exit(errno);
    }

    fprintf(outfile, "LSH\n");

    double total_AF = 0, total_recall = 0, total_t_lsh = 0, total_t_true = 0; //variables for counting averages

    for (int qi = 0; qi < sift_queries.number_of_vectors; ++qi) {
        std::vector<double> query_vec(sift.v_dim);
        for (int j = 0; j < sift.v_dim; ++j)
            query_vec[j] = static_cast<double>(sift_queries.vectors[qi][j]);   //sift data is raw so we dont normalize

        fprintf(outfile, "Query: %d\n", qi);
        //counting how long a sift vector takes into lsh 
        auto t1 = std::chrono::high_resolution_clock::now();
        auto results = lsh->knn_search(query_vec, sift.vectors, sift.number_of_vectors, p->n); //function to return n neighbors of query vector in lsh
        auto t2 = std::chrono::high_resolution_clock::now();
        double t_lsh = std::chrono::duration<double>(t2 - t1).count();
        total_t_lsh += t_lsh;
        //counting how long a sift vector takes into brute force
        auto t3 = std::chrono::high_resolution_clock::now();
        auto true_results = brute_force_knn_sift(query_vec, sift.vectors, p->n);  //function to return n neighbors of query vector in brute force
        auto t4 = std::chrono::high_resolution_clock::now();
        double t_true = std::chrono::duration<double>(t4 - t3).count();
        total_t_true += t_true;

        int hits = 0;
        double sum_AF = 0;

        for (int ni = 0; ni < p->n; ++ni) {
            int idx_approx = results[ni].first;                //getting the results for lsh
            double dist_approx = results[ni].second;
            int idx_true = true_results[ni].first;             //getting the results for brute force
            double dist_true = true_results[ni].second;

            fprintf(outfile, "Nearest neighbor-%d: %d\n", ni + 1, idx_approx);
            fprintf(outfile, "Nearest neighbor in brute-%d: %d\n", ni + 1, idx_true);
            fprintf(outfile, "distanceApproximate: %.6f\n", dist_approx);
            fprintf(outfile, "distanceTrue: %.6f\n", dist_true);

            if (dist_true != 0) {
                sum_AF += dist_approx / dist_true;          //special case where query vector was a part of the dataset
            } else {
                sum_AF += 1;
            }

            for (int k = 0; k < p->n; ++k)
                if (results[ni].first == true_results[k].first)         //seeing how many neighbors lsh found correctly compared to brute force
                    hits++;
        }
        // this part is where the range search happens if it has to
        if (p->range) {
            auto range_results = lsh->range_search_sift(query_vec, sift.vectors, sift.number_of_vectors, p->r);
            fprintf(outfile, "R-near neighbors:\n");
            for (auto& [idx, dist] : range_results)
                fprintf(outfile, "Image_ID: %d\n", idx);
        }

        double AF = sum_AF / p->n;
        double recall = static_cast<double>(hits) / p->n;
        total_AF += AF;
        total_recall += recall;

        fprintf(outfile, "Average AF: %.6f\n", AF);
        fprintf(outfile, "Recall@N: %.6f\n", recall);
        fprintf(outfile, "QPS: %.6f\n", 1.0 / t_lsh);                   //printing the averages for every query vector (optional)
        fprintf(outfile, "tApproximateAverage: %.6f\n", t_lsh);
        fprintf(outfile, "tTrueAverage: %.6f\n", t_true);
        fprintf(outfile, "\n");

        std::cout << "Query " << qi + 1 << " done \n"; 

        

        // if (qi == 99) {
        //     break;              //this keeps the query vectors to 100 so we can use the dataset but not run the entire query dataset
        // }
    }

    // if you want to run the entire query dataset delete if statement in lines 301-303 and line 309
    std::cout << "sift Queries =" << sift_queries.number_of_vectors << "\n";
    int nq = sift_queries.number_of_vectors;
    // nq = 100;  // Limit to 100 queries
    fprintf(outfile, "\n=== Global averages ===\n");
    fprintf(outfile, "Average AF: %.6f\n", total_AF / nq);
    fprintf(outfile, "Average Recall@N: %.6f\n", total_recall / nq);            //printing the averages (mandatory)
    fprintf(outfile, "Average QPS: %.6f\n", nq / total_t_lsh);
    fprintf(outfile, "Average tApproximate: %.6f\n", total_t_lsh / nq);
    fprintf(outfile, "Average tTrue: %.6f\n", total_t_true / nq);

    fclose(outfile);  // Close the output file

    delete lsh;
}


// ===========================================================================================================================
// 🔹 HYPERCUBE IMPLEMENTATION
// ===========================================================================================================================

// =========================================
// 🔹 MNIST Experiment Function
// =========================================

void run_mnist_experiment_hypercube(Params* p, MNISTData& mnist) {
    std::cout << "MNIST dataset loaded with " << mnist.number_of_images << " images.\n";

    // Convert MNIST data to doubles
    std::vector<std::vector<double>> mnist_dataset(mnist.number_of_images, std::vector<double>(mnist.image_size));
    for (int i = 0; i < mnist.number_of_images; ++i)
        for (int j = 0; j < mnist.image_size; ++j)
            mnist_dataset[i][j] = static_cast<double>(mnist.images[i][j]) / 255.0;  // Normalizing each pixel for mnist data

    // Create Hypercube dynamically
    Hypercube* cube = nullptr;
    cube = new Hypercube(p->kproj, mnist.image_size, p->w, p->seed, p->m, p->probes);

    // Insert all MNIST images into hypercube
    for (int i = 0; i < mnist.number_of_images; ++i)
        cube->insert(i, mnist_dataset[i]);

    // cube->print_table_info(); //use it only if you want info on the hypercube

    // Load query set
    FILE* fq = fopen("query.dat", "r");
    if (!fq) { perror("Failed to open query.dat"); exit(errno); }
    MNISTData mnist_queries = readInputMnist(fq);
    fclose(fq);
    std::cout << "Loaded " << mnist_queries.number_of_images << " query vectors.\n";

    // Open output file 
    FILE* outfile = fopen(p->o.c_str(), "w");
    if (!outfile) {
        perror("Failed to open hypercube_results.txt");
        exit(errno);
    }
    
    fprintf(outfile, "Hypercube\n");

    double total_AF = 0.0;
    double total_recall = 0.0;              //variables for counting averages
    double total_t_hyper = 0.0;
    double total_t_true = 0.0;

    // Query each query vector
    for (int qi = 0; qi < mnist_queries.number_of_images; ++qi) {
        std::vector<double> query_vec(mnist.image_size);
        for (int j = 0; j < mnist.image_size; ++j)
            query_vec[j] = static_cast<double>(mnist_queries.images[qi][j]) / 255.0;  // Normalizing query image for mnist query vectors

        fprintf(outfile, "Query: %d\n", qi);

        // timing hypercube search 
        auto t1 = std::chrono::high_resolution_clock::now();
        auto candidates = cube->query(query_vec, p->m);
        
        auto approx_results = cube->find_top_n_neighbors(query_vec, candidates, p->n, mnist_dataset);
        auto t2 = std::chrono::high_resolution_clock::now();
        double t_hyper = std::chrono::duration<double>(t2 - t1).count();
        total_t_hyper += t_hyper;
        // timing bruteforce search 
        auto t3 = std::chrono::high_resolution_clock::now();
        auto true_results = brute_force_knn(query_vec, mnist_dataset, p->n);
        auto t4 = std::chrono::high_resolution_clock::now();
        double t_true = std::chrono::duration<double>(t4 - t3).count();
        total_t_true += t_true;

        // --- Compute metrics ---
        int hits = 0;
        double sum_AF = 0.0;

        for (int ni = 0; ni < p->n; ++ni) {

            if (ni >= (int)approx_results.size()) {
                fprintf(outfile, "No further neighbors were found in approx_results.\n");
                break;
            }

            int idx_approx = approx_results[ni].index;

            double dist_approx = euclidean_distance(query_vec, mnist_dataset[idx_approx]);
                                                                                                            //getting the results of hyperqube and bruteforce
            int idx_true = true_results[ni].first;
            double dist_true = true_results[ni].second;

            fprintf(outfile, "Nearest neighbor-%d: %d\n", ni + 1, idx_approx);
            fprintf(outfile, "Nearest neighbor in brute-%d: %d\n", ni + 1, idx_true);
            fprintf(outfile, "distanceApproximate: %.6f\n", dist_approx);
            fprintf(outfile, "distanceTrue: %.6f\n", dist_true);

            if (dist_true != 0) {
                sum_AF += dist_approx / dist_true;          //special case where query vector was a part of the dataset
            } else {
                sum_AF += 1;
            }

            for (int k = 0; k < p->n; ++k)
                if (idx_approx == true_results[k].first)        //how many vectors hyperqube got right
                    hits++;
        }
        //range search part , if it has to happen
        if (p->range) {
            auto range_results = cube->range_search(query_vec, p->r, p->m, mnist_dataset);
            fprintf(outfile, "R-near neighbors:\n");
            for (const auto& res : range_results) {                                         
                fprintf(outfile, "Image_ID: %d\n", res.index);
            }
        }

        double AF = sum_AF / p->n;
        double recall = static_cast<double>(hits) / p->n;
        total_AF += AF;
        total_recall += recall;

        fprintf(outfile, "Average AF: %.6f\n", AF);
        fprintf(outfile, "Recall@N: %.6f\n", recall);
        fprintf(outfile, "QPS: %.6f\n", 1.0 / t_hyper);                 //printing the averages for every query vector (optional)
        fprintf(outfile, "tApproximateAverage: %.6f\n", t_hyper);
        fprintf(outfile, "tTrueAverage: %.6f\n", t_true);
        fprintf(outfile, "\n");

        std::cout << "Query " << qi + 1 << " done.\n";

        // if (qi == 99) {
        //     break;                              //like in lsh we use a subdataset of 100 vectors if you want to use all of it delete the if statement and the line 457
        // }
    }
    fprintf(outfile, "\n");

    int nq = mnist_queries.number_of_images;
    // nq = 100;
    fprintf(outfile, "\n=== Global averages over all queries ===\n");
    fprintf(outfile, "Average AF: %.6f\n", total_AF / nq);
    fprintf(outfile, "Average Recall@N: %.6f\n", total_recall / nq);
    fprintf(outfile, "Average QPS: %.6f\n", nq / total_t_hyper);                    //printing the averages (mandatory)
    fprintf(outfile, "Average tApproximate: %.6f\n", total_t_hyper / nq);
    fprintf(outfile, "Average tTrue: %.6f\n", total_t_true / nq);

    fclose(outfile);  // Close the output file
    delete cube;
}



// =========================================
// 🔹 SIFT Experiment Function
// =========================================

void run_sift_experiment_hypercube(Params* p, SIFTData& sift) {
    std::cout << "SIFT dataset loaded with " << sift.number_of_vectors << " images.\n";

    // Convert sift data to doubles
    std::vector<std::vector<double>> sift_dataset(sift.number_of_vectors, std::vector<double>(sift.v_dim));
    for (int i = 0; i < sift.number_of_vectors; ++i)
        for (int j = 0; j < sift.v_dim; ++j)
            sift_dataset[i][j] = static_cast<double>(sift.vectors[i][j]);           //sift data are raw so no normalization

    // Create Hypercube dynamically
    Hypercube* cube = nullptr;
    cube = new Hypercube(p->kproj, sift.v_dim, p->w, p->seed, p->m, p->probes);

    // Insert all sift images into hypercube
    for (int i = 0; i < sift.number_of_vectors; ++i)
        cube->insert(i, sift_dataset[i]);

    // cube->print_table_info(); //use it only if you want info on the hypercube

    // Load query set
    FILE* fq = fopen("query.dat", "r");
    if (!fq) { perror("Failed to open query.dat"); exit(errno); }
    SIFTData sift_queries = readInputSift2(fq);
    fclose(fq);
    std::cout << "Loaded " << sift_queries.number_of_vectors << " query vectors.\n";

    // Open output 
    FILE* outfile = fopen(p->o.c_str(), "w");
    if (!outfile) {
        perror("Failed to open hypercube_results.txt");
        exit(errno);
    }
    
    fprintf(outfile, "Hypercube\n");

    double total_AF = 0.0;
    double total_recall = 0.0;
    double total_t_hyper = 0.0;
    double total_t_true = 0.0;

    // Query each query vector
    for (int qi = 0; qi < sift_queries.number_of_vectors; ++qi) {
        std::vector<double> query_vec(sift.v_dim);
        for (int j = 0; j < sift.v_dim; ++j)
            query_vec[j] = static_cast<double>(sift_queries.vectors[qi][j]);  //sift data so no normalization

        fprintf(outfile, "Query: %d\n", qi);

        // timing hypercube search 
        auto t1 = std::chrono::high_resolution_clock::now();
        auto candidates = cube->query(query_vec, p->m);

        auto approx_results = cube->find_top_n_neighbors(query_vec, candidates, p->n, sift_dataset);
        auto t2 = std::chrono::high_resolution_clock::now();
        double t_hyper = std::chrono::duration<double>(t2 - t1).count();
        total_t_hyper += t_hyper;
        
        // timing brute force search 
        auto t3 = std::chrono::high_resolution_clock::now();
        auto true_results = brute_force_knn_sift(query_vec, sift.vectors, p->n);
        auto t4 = std::chrono::high_resolution_clock::now();
        double t_true = std::chrono::duration<double>(t4 - t3).count();
        total_t_true += t_true;

        // --- Compute metrics ---
        int hits = 0;
        double sum_AF = 0.0;

        for (int ni = 0; ni < p->n; ++ni) {

            if (ni >= (int)approx_results.size()) {
                fprintf(outfile, "No further neighbors were found in approx_results.\n");
                break;
            }

            int idx_approx = approx_results[ni].index;

            double dist_approx = euclidean_distance(query_vec, sift_dataset[idx_approx]);
                                                                                                            //getting the results of hyperqube and bruteforce
            int idx_true = true_results[ni].first;
            double dist_true = true_results[ni].second;

            fprintf(outfile, "Nearest neighbor-%d: %d\n", ni + 1, idx_approx);
            fprintf(outfile, "Nearest neighbor in brute-%d: %d\n", ni + 1, idx_true);
            fprintf(outfile, "distanceApproximate: %.6f\n", dist_approx);
            fprintf(outfile, "distanceTrue: %.6f\n", dist_true);

            if (dist_true != 0) {
                sum_AF += dist_approx / dist_true;                                  //special case where query vector was a part of the dataset
            } else {
                sum_AF += 1;
            }

            for (int k = 0; k < p->n; ++k)
                if (idx_approx == true_results[k].first)                //how many neighbors hyperqube got right
                    hits++;
        }
        //possible range search (if necessary)
        if (p->range) {
            auto range_results = cube->range_search_sift(query_vec, p->r, p->m, sift_dataset);
            fprintf(outfile, "R-near neighbors:\n");
            for (const auto& res : range_results) {
                fprintf(outfile, "Image_ID: %d\n", res.index);
            }
        }

        double AF = sum_AF / p->n;
        double recall = static_cast<double>(hits) / p->n;
        total_AF += AF;
        total_recall += recall;

        fprintf(outfile, "Average AF: %.6f\n", AF);
        fprintf(outfile, "Recall@N: %.6f\n", recall);
        fprintf(outfile, "QPS: %.6f\n", 1.0 / t_hyper);
        fprintf(outfile, "tApproximateAverage: %.6f\n", t_hyper);                           //printing the averages for every query vector (optional)
        fprintf(outfile, "tTrueAverage: %.6f\n", t_true);
        fprintf(outfile, "\n");

        std::cout << "Query " << qi + 1 << " done.\n";

        // if (qi == 99) {                                                         //like in lsh we use a subdataset of 100 vectors if you want to use all of it delete the if statement and the line 602
        //     break;
        // }
    }
    fprintf(outfile, "\n");

    int nq = sift_queries.number_of_vectors;
    // nq = 100;
    fprintf(outfile, "\n=== Global averages over all queries ===\n");
    fprintf(outfile, "Average AF: %.6f\n", total_AF / nq);
    fprintf(outfile, "Average Recall@N: %.6f\n", total_recall / nq);
    fprintf(outfile, "Average QPS: %.6f\n", nq / total_t_hyper);                                //printing the averages (mandatory)
    fprintf(outfile, "Average tApproximate: %.6f\n", total_t_hyper / nq);
    fprintf(outfile, "Average tTrue: %.6f\n", total_t_true / nq);

    fclose(outfile);  // Close the output file
    delete cube;
}

