import sys
import numpy as np
import time
import torch
import torch.nn.functional as F
from data_parser import read_sift, read_mnist , SearchParser
from models import MLPClassifier
from nlsh_build import get_choice



# -----------------------------------------------------------
#              UTILITY FUNCTIONS
# -----------------------------------------------------------

#this simply returns the euclidean distance (L2 distance) between vector a and vector b
def euclidean(a, b):
    return np.linalg.norm(a - b)

#this function uses brute force to find the closest vector to the query vector and also returns the complete lists of distances of the query vector to every vector of the dataset
def brute_force_search(query, X):
    dists = np.linalg.norm(X - query, axis=1) # finds the distances
    idx = np.argmin(dists) # finds which index has the smallest distance
    return idx, dists[idx], dists   # returns closest vector to the query vector and the complete lists of distances of the query vector to every vector of the dataset

#this function returns the top T bins to reduce the search
def top_t_bins(model, query, T):
    model.eval()  # ensure model is in evaluation mode
    with torch.no_grad():
        q = torch.tensor(query, dtype=torch.float32).unsqueeze(0)  # shape: (1, d)
        logits = model(q)  # shape: (1, m)
        probs = torch.softmax(logits, dim=1)  # softmax over bins      

        top_bins = torch.topk(probs, T, dim=1).indices.squeeze(0)  # this finds top T bin indices


        return top_bins.tolist()


#this function searches inside the top T bins and finds the N nearest ones
def search_in_bins(query, X, inverted_lists, bins, N):
    
    best = [] #create empty list

    for b in bins:
        for idx in inverted_lists[b]:   #checks the vectors inside the inverted lists , finds the euclidean distance and keeps the best (the ones closer to query vector)
            d = euclidean(query, X[idx])
            best.append((d, idx))

    best.sort()
    return best[:N]    # returns N nearest vectors






#this function simply loads the index that was created by nlsh_build 
def load_index(path):
    index = torch.load(path, map_location="cpu")

    model = MLPClassifier(
        d_in=index["dimension"],
        n_out=index["m"],
        hidden_dim=index["nodes"],
        num_layers=index["layers"],
        dropout=index["dropout"],
        batchnorm=index["batchnorm"]
    )

    model.load_state_dict(index["model_state"])
    model.eval()

    blocks = index["blocks"]
    inverted_lists = index["inverted_lists"]

    metadata = {
        "m": index["m"],
        "dimension": index["dimension"],
        "knn": index["knn"],
        "seed": index["seed"],
        "layers": index["layers"],
        "nodes": index["nodes"],
        "dropout": index["dropout"],
        "batchnorm": index["batchnorm"],
        "kahip_mode": index["kahip_mode"],
        "imbalance": index["imbalance"]
    }

    return model, blocks, inverted_lists, metadata


# -----------------------------------------------------------
#                 MAIN SEARCH FUNCTION
# -----------------------------------------------------------

def nlsh_search(p):

    # -------------------------------------------------------
    # Load dataset (MNIST or SIFT)
    # -------------------------------------------------------
    print("[INFO] Loading dataset...")

    if p.type == "mnist":
        data = read_mnist(p.input)
        X = np.array(data.images, dtype=np.float32)
        q_data = read_mnist(p.query)
        Q = np.array(q_data.images, dtype=np.float32)

        X = X / 255.0   #data normalization
        Q = Q / 255.0        
    else:
        data = read_sift(p.input)
        X = np.array(data.dataset, dtype=np.float32)
        q_data = read_sift(p.query)
        Q = np.array(q_data.dataset, dtype=np.float32)




    choice = get_choice()

    if choice == "1":
        DEBUG_X = 5000  # If brute-force, use a debug subset because brute force is extremely slow for the actual datasets. DEBUG_X has to be the same in nlsh_build and nlsh_search
        X = X[:DEBUG_X]
        print(f"[DEBUG] Using only {len(X)} vectors for search")


    model, blocks, inverted_lists, metadata = load_index(p.path)    #loads the index
    model.eval()

    fout = open(p.output, "w")

    t_apx_total = 0         #creation of initial variables to track progress
    t_true_total = 0
    AF_sum = 0
    recall_hits = 0


    # -------------------------------------------------------
    # QUERY PROCESSING LOOP
    # -------------------------------------------------------
    for qid, q in enumerate(Q):

        fout.write(f"Query: {qid}\n\n")

        # -----------------------------------------
        # TRUE NEAREST NEIGHBOR (BRUTE FORCE)
        # -----------------------------------------
        t0 = time.time()
        true_idx, true_dist, all_true_dists = brute_force_search(q, X)
        t_true_total += time.time() - t0

        # -----------------------------------------
        # APPROXIMATE LSH SEARCH
        # -----------------------------------------
        t1 = time.time()
        bins = top_t_bins(model, q, p.T)
        approx = search_in_bins(q, X, inverted_lists, bins, p.N)
        t_apx_total += time.time() - t1

        # -----------------------------------------
        # PRINT TOP-N NEIGHBORS
        # -----------------------------------------
        for rank, (apx_dist, apx_idx) in enumerate(approx, start=1):

            fout.write(f"Nearest neighbor-{rank}: {apx_idx}\n")
            fout.write(f"distanceApproximate: {apx_dist}\n")
            fout.write(f"distanceTrue: {all_true_dists[apx_idx]}\n\n")

        # -----------------------------------------
        # Range Search for R-NEAR NEIGHBORS (conditional)
        # -----------------------------------------
        if p.range:

            R_bins = top_t_bins(model, q, p.T)
            R_approx = search_in_bins(q, X, inverted_lists, R_bins, 30) #max of 30 so the output file doesnt become hard to read , if you want more broad range search simply change the number 30 to whatever you want

            # Normalize R depending on dataset
            if p.type == "mnist":
                R_norm = p.R / 255.0
            else:  # SIFT
                R_norm = p.R / 218.0

            fout.write("R-near neighbors:\n")

            for apx_dist, apx_idx in R_approx:    # write all the neighbors inside the radius
                if apx_dist <= R_norm:          # we use nlsh to find the neighbors with a max of 30 so the output file doesnt become hard to read 
                    fout.write(f"{apx_idx}\n")

            fout.write("\n")


        # -----------------------------------------
        # METRICS
        # -----------------------------------------
        if true_dist == 0:
            AF_sum += 1   # this in the event a vector exists inside the input and query
        else:
            AF_sum += approx[0][0] / true_dist


        approx_ids = [idx for (_, idx) in approx]
        if true_idx in approx_ids:              #finds how many correct vectors nlsh has found
            recall_hits += 1

        if qid == 999:      #this is so the user doesnt wait for the entire query dataset to see the results ,if you want to see the full results of the query instead of a sample 
            break           # comment this if statement and line 208 and uncomment the line 207

    # total_queries = len(Q)
    total_queries = 1000        #how many vector queries results would you like to appear on the output file

    fout.write(f"Average AF: {AF_sum / total_queries}\n")
    fout.write(f"Recall@N: {recall_hits / total_queries}\n")
    fout.write(f"QPS: {total_queries / t_apx_total}\n")                     #printing the final results
    fout.write(f"tApproximateAverage: {t_apx_total / total_queries}\n")
    fout.write(f"tTrueAverage: {t_true_total / total_queries}\n")

    fout.close()
    print("[OK] Finished. Output file:", p.output)





# -----------------------------------------------------------
#                       MAIN
# -----------------------------------------------------------

def main():
    p = SearchParser(sys.argv)

    print("====================================")
    print("          NLSH SEARCH PROGRAM        ")
    print("====================================")
    print(f"Dataset: {p.input}")
    print(f"Query: {p.query}")
    print(f"Index: {p.path}")
    print(f"Output: {p.output}")
    print(f"N: {p.N}")
    print(f"T: {p.T}")
    print(f"Range search: {p.range}")
    print("====================================\n")

    nlsh_search(p)




if __name__ == "__main__":
    main()
