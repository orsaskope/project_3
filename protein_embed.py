import torch
import esm
import os
import numpy as np
from Bio import SeqIO
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--ids", default="input_ids.txt")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    # --- STEP 1: Embedding Generation (Checking if files exist first) ---
    # no need if we already have the data
    if os.path.exists(args.output) and os.path.exists(args.ids):
        print(f"Files '{args.output}' and '{args.ids}' already exist. Skipping embedding generation.")
    else:
        print("Starting Embedding Generation (Class Style)...")
        # Loading the pretrained ESM2 model
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval() # Set to evaluation mode (no dropout etc.)
        
        # Check if we have a GPU available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        D = 320 #dimension
        fvec = None
        fid = None

        try:
            # Opening files: binary for vectors, text for IDs
            fvec = open(args.output, "wb")
            fid = open(args.ids, "w")
            batch_data = []
            count = 0

            for record in SeqIO.parse(args.input, "fasta"):
                seq = str(record.seq)
                # ESM has a limit of 1022 aminoacids
                if len(seq) > 1022:
                    seq = seq[:1022] 
                
                batch_data.append((record.id, seq))
                # Proccesing sequences in batches to speed up things
                if len(batch_data) == args.batch:
                    labels, strs, tokens = batch_converter(batch_data)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        # Extracting representations from the 6th layer
                        results = model(tokens, repr_layers=[6], return_contacts=False)
                    
                    token_embeddings = results["representations"][6]
                    for i in range(len(batch_data)):
                        # Calculate mean embedding
                        embedding = token_embeddings[i, 1 : len(strs[i]) + 1].mean(dim=0)
                        emb_np = embedding.cpu().numpy().astype(np.float32)
                        
                        # Writing in the specific format: [Dimension][Vector_Data]
                        fvec.write(np.int32(D).tobytes())
                        fvec.write(emb_np.tobytes())
                        fid.write(labels[i] + "\n")
                    
                    count += len(batch_data)
                    print(f"Processed {count} sequences...")
                    batch_data = []

            if batch_data: # last batch 
                labels, strs, tokens = batch_converter(batch_data)
                tokens = tokens.to(device)
                with torch.no_grad():
                    results = model(tokens, repr_layers=[6])
                token_embeddings = results["representations"][6]
                for i in range(len(batch_data)):
                    embedding = token_embeddings[i, 1 : len(strs[i]) + 1].mean(dim=0)
                    emb_np = embedding.cpu().numpy().astype(np.float32)
                    fvec.write(np.int32(D).tobytes())
                    fvec.write(emb_np.tobytes())
                    fid.write(labels[i] + "\n")
            
            print(f"Embedding generation completed.")

        finally:
            #closing the files
            if fvec: fvec.close()
            if fid: fid.close()

    # --- STEP 2: Neural LSH Build Phase ---
    # We check if the index and folders already exist to avoid retraining the model aor continuing the programme
    index_name = "nlsh_index"
    folder_path = "project_2-main"
    index_full_path = os.path.join(folder_path, index_name)

    

    if os.path.exists(index_full_path):
        print(f"\nNeural LSH Index '{index_name}' already exists. Skipping Build Phase.")
    else:
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found!")
            return

        print(f"\nStarting Neural LSH Build Phase...")
        absolute_input_path = os.path.abspath(args.output)
        
        build_command = [
            "python3", "nlsh_build.py",
            "-d", absolute_input_path, 
            "-i", index_name,
            "-type", "sift",
            "--knn", "15",
            "-m", "100",
            "--epochs", "10",
            "--nodes", "256"
        ]

        try:
            print(f"Executing nlsh_build.py inside folder: {folder_path}")
            subprocess.run(
                build_command, 
                cwd=folder_path, 
                input=b"2\n", # Automatically sending '2' so when the script asks for input it chooses ivfflat
                check=True
            )
            print("Neural LSH Build Phase completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during nlsh_build: {e}")

if __name__ == "__main__":
    main()