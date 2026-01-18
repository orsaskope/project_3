import sys
import numpy as np
import kahip
import subprocess
import os
from data_parser import BuildParser, read_mnist, read_sift
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from graph_utils import compute_knn, build_weighted_knn_graph, knn_graph_to_csr, bwg_ivfflat, build_csr_ivfflat
from models import MLPClassifier, build_dataloader, train_model, get_device



# ==========================================================
#                     HELPER FUNCTIONS
# ==========================================================

# returns user choice between brute force or ivfflat
def get_choice():
    choice = ""
    while choice != "1" and choice != "2":
        print("Choose kNN graph construction method:")
        print(" 1) Brute Force (Python)")
        print(" 2) IVFFLAT C++ program\n")
        choice = input("Enter option (1 or 2): ").strip()
        if choice != "1" and choice != "2":
            print("Invalid choice. Please enter 1 or 2.\n")
    return choice


def load_dataset(p, choice):
    if p.type == "mnist":
        data = read_mnist(p.input)
        X = np.array(data.images, dtype=np.float32)
        N = data.number_of_images
        print(f"Loaded MNIST: {len(X)} vectors (full N = {N})")
        X = X / 255.0   #normalization of the data
    else:
        data = read_sift(p.input)
        X = np.array(data.dataset, dtype=np.float32)
        N = data.count
        print(f"Loaded SIFT: {len(X)} vectors (full N = {N})")
        X = X / 218.0   #normalization of the data

    # If brute-force, use a debug subset because brute force is extremely slow for the actual datasets.
    if choice == "1":
        DEBUG_X = 5000
        X = X[:DEBUG_X]
        print(f"[DEBUG] Using only {len(X)} vectors for brute-force")

    return X, N


#in case of brute force we call the 3 functions in order to create CSR
def build_knn_bruteforce(X, p):
    
    print("\n[1] Computing brute-force kNN graph...")
    knn_neighbors = compute_knn(X, p.knn)

    print("\n[2] Building weighted kNN graph...")
    w_knn_graph = build_weighted_knn_graph(knn_neighbors)

    print("\n[3] Converting weighted kNN graph to CSR arrays...")
    return knn_graph_to_csr(w_knn_graph)

#here we call the infflat subprocess if tmp.txt doesnt already exists
def build_knn_ivfflat(p, N):

    print("\n[1] Running IVFFLAT C++ search program...")

    exec_path = "./search"
    output_path = "./tmp.txt"

    if os.path.exists(output_path):
        print("[INFO] tmp.txt already exists → skipping IVFFLAT subprocess.")
    else:
        print("[INFO] tmp.txt does not exist → generating it now...")
        result = subprocess.run(
            [
                exec_path,
                "-ivfflat",
                "-type", p.type,
                "-seed", str(p.seed),
                "-d", p.input,
                "-kclusters", "64",
                "-range", "false",
                "-N", str(p.knn),
                "-o", output_path,
                "-nprobe", "4",
                "-R", "500"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    print(result.stdout)

    print("\n[2] Building weighted graph from IVFFLAT output...")
    w_knn_graph = bwg_ivfflat(output_path, N)

    print("\n[3] Converting graph to CSR arrays...")
    return build_csr_ivfflat(w_knn_graph, N)



#this function simply calls kahip and returns the results (edgecut,blocks)
def run_kahip(p, vwgt, xadj, adjcwgt, adjncy):

    print("\n[4] Running KaHIP partitioner...")

    edgecut, blocks = kahip.kaffpa(
        vwgt, xadj, adjcwgt, adjncy,
        p.m,            # number of partitions
        p.imbalance,    # imbalance
        1,              # suppress output
        p.seed,         # seed
        p.kahip_mode    # mode
    )

    return edgecut, blocks


#this function trains the model by 
#
#1)building the datalaoder
#2)creating the model
#3)training it using the train model function

def train_mlp(X, blocks, p):

    print("\n[5] Training Neural LSH MLP model...")

    # Build dataloader using correct batch size
    loader = build_dataloader(X, blocks, batch_size=p.batch_size)

    # Build model using ALL arguments from p
    model = MLPClassifier(
        d_in=X.shape[1],
        n_out=p.m,
        hidden_dim=p.nodes,
        num_layers=p.layers,
        dropout=p.dropout,
        batchnorm=p.batchnorm
    )

    # Train model using learning rate and epochs
    train_model(
        model=model,
        loader=loader,
        epochs=p.epochs,
        lr=p.lr
    )

    print("\n[OK] Training finished.")
    return model

#this function simply saves the index created with all the necessary nlsh information (KaHIP partition labels,inverted lists,Metadata)
def save_index(path, model, blocks, inverted_lists, p, dim):

    index = {
        "model_state": model.state_dict(),

        # KaHIP partition labels
        "blocks": np.array(blocks, dtype=np.int32),

        # The actual inverted index
        "inverted_lists": inverted_lists,

        # --- Metadata ---
        "m": p.m,
        "dimension": dim,
        "seed": p.seed,
        "layers": p.layers,
        "nodes": p.nodes,
        "dropout": p.dropout,
        "batchnorm": p.batchnorm,
        "knn": p.knn,
        "kahip_mode": p.kahip_mode,
        "imbalance": p.imbalance,
    }


    torch.save(index, path)    # save index to path given by user
    print(f"\n[OK] Index saved to {path}")

#this function creates an inverted list showing how many vectors each block has
def build_inverted_lists(blocks, m):

    inverted = [[] for _ in range(m)] #create empty lists for all blocks

    for idx, block in enumerate(blocks): #fill them with the vector ids
        inverted[block].append(idx)

    return inverted

# ==========================================================
#                     MAIN PIPELINE
# ==========================================================
def main():
    p = BuildParser(sys.argv)

    print("======================================")
    print("          NLSH BUILD PROGRAM          ")
    print("======================================")
    print(f"Dataset       : {p.input}")
    print(f"Index Path    : {p.index_path}")
    print(f"Type          : {p.type}")
    print(f"kNN           : {p.knn}")
    print(f"m (clusters)  : {p.m}")
    print(f"epochs        : {p.epochs}")
    print("======================================\n")

    # Step 1: user chooses kNN method
    choice = get_choice()

    # Step 2: load dataset based on choice , followed by creation of CSR depending on the choice of the user
    X, N = load_dataset(p, choice)

    if choice == "1":
        xadj, adjncy, adjcwgt, vwgt = build_knn_bruteforce(X, p)
    else:
        xadj, adjncy, adjcwgt, vwgt = build_knn_ivfflat(p, N)


    # Step 3: KaHIP partitioning
    edgecut, blocks = run_kahip(p, vwgt, xadj, adjcwgt, adjncy)


    # Step 4: training of the model
    model = train_mlp(X, blocks, p)

    # Step 5: build inverted lists
    inverted_lists = build_inverted_lists(blocks, p.m)

    # Step 6: Save index
    save_index(p.index_path, model, blocks, inverted_lists, p, dim=X.shape[1])


    print("Full NLSH pipeline completed!")



if __name__ == "__main__":
    main()
