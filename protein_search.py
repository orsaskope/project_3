import os
import torch
import esm
import subprocess
import argparse
import numpy as np
import pandas as pd
import shutil
from Bio import SeqIO

# Global constants
MODEL_NAME = "esm2_t6_8M_UR50D"
D = 320  
BATCH_SIZE = 32

def parse_output(file):
    """
    Reads the output file from the search algos and extracts neighbors 
    and performance metrics like Recall and QPS.
    """
    neighbors = {}
    stats = {
        "avg_recall": None,
        "avg_qps": None,
        "avg_af": None
    }
    current_query = None
    
    if not os.path.exists(lsh_file):
        print(f"Warning: File {lsh_file} not found.")
        return neighbors, stats

    with open(lsh_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        # 1. Capture Neighbors IDs
        if line.startswith("Query:"):
            current_query = line.split(":")[1].strip()
            neighbors[current_query] = []
        
        elif line.startswith("Nearest neighbor-") and "in brute" not in line:
            parts = line.split(":")
            if len(parts) > 1:
                neighbor_id = parts[1].strip()
                if current_query is not None:
                    neighbors[current_query].append(neighbor_id)

        # 2. Capture Stats 
        elif "Average Recall@N:" in line:
            stats["avg_recall"] = line.split(":")[1].strip()
        elif "Average QPS:" in line:
            stats["avg_qps"] = line.split(":")[1].strip()
        elif "Average tApproximate:" in line:
            stats["avg_tAp"] = line.split(":")[1].strip()

                
    return neighbors, stats

def process_blast_ground_truth(blast_file, e_value_threshold=0.001, n_top=50):
    """
    Proccesses the BLAST output to create a "ground Truth" dictionary.
    filters by E-value and keeps only the top n hits for each query.
    """
    column_names = [
        'query_id', 'subject_id', 'identity', 'alignment_length', 
        'mismatches', 'gap_opens', 'q_start', 'q_end', 
        's_start', 's_end', 'e_value', 'bit_score'
    ]
    
    if not os.path.exists(blast_file):
        print(f"Warning: Blast file {blast_file} not found.")
        return {}


    # Use pandas for fast filtering and grouping
    df = pd.read_csv(blast_file, sep='\t', names=column_names)
    filtered_df = df[df['e_value'] <= e_value_threshold].copy()
    
    # Sort by bit_score to ensure we get the best biological matches
    filtered_df = filtered_df.sort_values(by=['query_id', 'bit_score'], ascending=[True, False])
    
    ground_truth = {}
    grouped = filtered_df.groupby('query_id')
    
    for q_id, group in grouped:
        top_hits = group.head(n_top)
        # We use a set for fast intersection later
        ground_truth[q_id] = set(top_hits['subject_id'].unique())
        
    print(f"Processed {len(ground_truth)} queries from BLAST ground truth.")
    return ground_truth

def create_fasta_mapping(fasta_file):
    """
    Maps the index of a protein (0, 1, 2...) to its actual FASTA ID.
    Vital for comparing LSH results with BLAST.
    """
    mapping = {}
    if not os.path.exists(fasta_file):
        print(f"Warning: Fasta file {fasta_file} not found.")
        return mapping

    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        mapping[str(i)] = record.id 
    
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Protein Search Phase - Hardcoded Style")
    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-m", "--method", default="nlsh", choices=["lsh", "hypercube", "ivfflat", "ivfpq", "nlsh", "all"])
    
    args = parser.parse_args()
    query_vectors_path = "query.dat"

    # --- STEP 1: Embedding Generation for Queries ---
    if not os.path.exists(query_vectors_path):
        print(f"Generating query embeddings...")
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        with open(query_vectors_path, "wb") as fvec:
            batch_data = []
            for record in SeqIO.parse(args.query, "fasta"):
                seq = str(record.seq)[:1022] # ESM-2 limit is 1022 tokens
                batch_data.append((record.id, seq))
                if len(batch_data) == BATCH_SIZE:
                    labels, strs, tokens = batch_converter(batch_data)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        results = model(tokens, repr_layers=[6], return_contacts=False)
                    token_embeddings = results["representations"][6]
                    for i in range(len(batch_data)):
                        # Mean pooling to get a single vector per protein
                        embedding = token_embeddings[i, 1 : len(strs[i]) + 1].mean(dim=0)
                        fvec.write(np.int32(D).tobytes())
                        fvec.write(embedding.cpu().numpy().astype(np.float32).tobytes())
                    batch_data = []
            
            if batch_data: # Last batch
                labels, strs, tokens = batch_converter(batch_data)
                tokens = tokens.to(device)
                with torch.no_grad():
                    results = model(tokens, repr_layers=[6])
                for i in range(len(batch_data)):
                    embedding = results["representations"][6][i, 1 : len(strs[i]) + 1].mean(dim=0)
                    fvec.write(np.int32(D).tobytes())
                    fvec.write(embedding.cpu().numpy().astype(np.float32).tobytes())

    # Copy query vectors to project folders so the C++ programs can find them
    if os.path.exists("project_1-main"):
        shutil.copy(query_vectors_path, os.path.join("project_1-main", query_vectors_path))
    if os.path.exists("project_2-main"):
        shutil.copy(query_vectors_path, os.path.join("project_2-main", query_vectors_path))

    # --- STEP 2: Execution of Search Algorithms ---

    #in every section the subprocess happens 1) if its the only one the user wants or 
    #2)if the user chose "all" methods to be used 

    abs_data = os.path.abspath(args.data)
    abs_query = "query.dat"
    
    # LSH SECTION
    if args.method == "lsh" or args.method == "all":
        print("\n>>> Executing LSH Search...")
        folder_lsh = "project_1-main"
        cmd_lsh = [
            "./search",
            "-lsh",
            "-k", "4",
            "-L", "5",
            "-w", "4.0",
            "-d", abs_data,
            "-q", abs_query,
            "-o", "output_LSH.txt",
            "-type", "sift",
            "-range", "false",
            "-N", "2",
            "-R", "500"
        ]
        if os.path.exists(folder_lsh):
            subprocess.run(cmd_lsh, cwd=folder_lsh, input=b"2\n")

    # HYPERCUBE SECTION
    if args.method == "hypercube" or args.method == "all":
        print("\n>>> Executing Hypercube Search...")
        folder_cube = "project_1-main"
        cmd_cube = [
            "./search",
            "-hypercube",
            "-kproj", "16",
            "-w", "1.0",
            "-M", "5000",
            "-probes", "4",
            "-d", abs_data,
            "-q", abs_query,
            "-o", "output_hypercube.txt",
            "-type", "sift",
            "-range", "false",
            "-N", "5",
            "-R", "500"
        ]
        if os.path.exists(folder_cube):
            subprocess.run(cmd_cube, cwd=folder_cube, input=b"2\n")

    # IVFFLAT SECTION
    if args.method == "ivfflat" or args.method == "all":
        print("\n>>> Executing IVF-Flat Search...")
        folder_flat = "project_1-main"
        cmd_flat = [
            "./search",
            "-ivfflat",
            "-kclusters", "64",
            "-nprobe", "4",
            "-seed", "42",
            "-d", abs_data,
            "-q", abs_query,
            "-o", "output_ivfflat.txt",
            "-type", "sift",
            "-range", "false",
            "-N", "2",
            "-R", "500"
        ]
        if os.path.exists(folder_flat):
            subprocess.run(cmd_flat, cwd=folder_flat, input=b"2\n")

    # IVFPQ SECTION
    if args.method == "ivfpq" or args.method == "all":
        print("\n>>> Executing IVF-PQ Search...")
        folder_pq = "project_1-main"
        cmd_pq = [
            "./search",
            "-ivfpq",
            "-kclusters", "64",
            "-nprobe", "4",
            "-M", "10",
            "-nbits", "8",
            "-seed", "42",
            "-d", abs_data,
            "-q", abs_query,
            "-o", "output_ivfpq",
            "-type", "sift",
            "-range", "false",
            "-N", "2",
            "-R", "500"
        ]
        if os.path.exists(folder_pq):
            subprocess.run(cmd_pq, cwd=folder_pq, input=b"2\n")

    # NLSH SECTION
    if args.method == "nlsh" or args.method == "all":
        print("\n>>> Executing Neural LSH Search...")
        folder_nlsh = "project_2-main"
        cmd_nlsh = [
            "python3", 
            "nlsh_search.py",
            "-d", abs_data,
            "-q", abs_query,
            "-i", "nlsh_index",
            "-o", "output_NLSH.txt",
            "-type", "sift",
            "-N", "2",
            "-T", "10",
            "-range", "false"
        ]
        if os.path.exists(folder_nlsh):
            subprocess.run(cmd_nlsh, cwd=folder_nlsh, input=b"2\n")

    print("\nSearch Phase Finished.")

    # --- STEP 3: Evaluation Phase ---
    print("\n>>> Processing Ground Truth and Mapping...")
    # Ground truth from BLAST
    blast_gt = process_blast_ground_truth("blast_results.tsv")
    
    # These mappings are very important to translate indices to actual names
    # Proccesing mapping for the targets and queries
    target_map = create_fasta_mapping("swissprot.fasta")    
    query_map = create_fasta_mapping("targets.fasta")

    """
    We weren't able to complete the exercise however the next step would 
    be to use the contents obtained from the blast function and the parse_output 
    and compare them to create the results.txt.
    """



if __name__ == "__main__":
    main()