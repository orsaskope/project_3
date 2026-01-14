from transformers import EsmTokenizer, EsmModel
from Bio import SeqIO
import argparse
import numpy as np
import torch

MODEL = "facebook/esm2_t6_8M_UR50D"
FASTA_PATH = "swissprot_50k.fasta"

BATCH_SIZE = 64
MAX_LEN = 512


# COmpute embeddings for a sequence batch
def run_batch(model, tokenizer, seqs, device, max_len):
    inputs = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state                 
        mask = inputs["attention_mask"].unsqueeze(-1)      
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1) # (B, D)

    return emb.detach().cpu().numpy().astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output vectors file (fvecs)")
    parser.add_argument("--ids", default="input_ids.txt", help="Output IDs file")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length for truncation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = EsmTokenizer.from_pretrained(MODEL)
    model = EsmModel.from_pretrained(MODEL).to(device)
    model.eval()

    fvec = open(args.output, "wb")
    fid = open(args.ids, "w")

    count = 0
    batch_ids = []
    batch_seqs = []
    D = None

    try:
        # Read fasta file in batches
        for record in SeqIO.parse(args.input, "fasta"):
            count += 1
            batch_ids.append(record.id)
            batch_seqs.append(str(record.seq))

            if count % 200 == 0:
                print("Read sequences:", count)

            if len(batch_seqs) == args.batch:
                E = run_batch(model, tokenizer, batch_seqs, device, args.max_len)  # (B, D)

                if D is None:
                    D = E.shape[1]
                    print("Embedding dim D =", D)

                # batch in SIFT fvecs: [int32 D][float32*D]...
                for i in range(E.shape[0]):
                    fvec.write(np.int32(D).tobytes())
                    fvec.write(E[i].tobytes())
                    fid.write(batch_ids[i] + "\n")
                fvec.flush()
                fid.flush()

                batch_ids, batch_seqs = [], []

                if count % 1000 == 0:
                    print("Processed:", count)

        # last batch
        if batch_seqs:
            E = run_batch(model, tokenizer, batch_seqs, device, args.max_len)

            if D is None:
                D = E.shape[1]
                print("Embedding dim D =", D)

            for i in range(E.shape[0]):
                fvec.write(np.int32(D).tobytes())
                fvec.write(E[i].tobytes())
                fid.write(batch_ids[i] + "\n")

    finally:
        fvec.close()
        fid.close()

    print("DONE. Total sequences processed:", count)


if __name__ == "__main__":
    main()