import argparse
import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from bmfm_sm_gen.data.dm_mol import MolecularDataModule
from bmfm_sm_gen.launch.preprocess_data import smiles_to_selfies
from bmfm_sm_gen.trainers.lm_vae import VAELightningModule


def encode_selfies_list(selfies_iter, tokenizer, max_len, device, limit):
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for idx, selfies_str in enumerate(selfies_iter):
            if selfies_str is None or selfies_str == "":
                continue
            tok = tokenizer(
                selfies_str,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )[
                "input_ids"
            ]  # shape: (1, L)
            src = tok[:-1].to(device)  # drop last token to match training shift
            # encode -> z
            mu, logvar = lightning_module.model.encode(src.unsqueeze(0))
            z = lightning_module.model.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu().numpy())
            labels.append(0)
            if len(labels) >= limit:
                break
    return latent_vectors, labels


def main():
    parser = argparse.ArgumentParser(description="Latent Space Analysis")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--selfies_csv",
        type=str,
        default=None,
        help="Path to CSV of SELFIES sampled earlier (uses column --selfies_column). If set, overrides data_dir.",
    )
    parser.add_argument(
        "--selfies_column",
        type=str,
        default="selfies",
        help="Column name in --selfies_csv containing SELFIES strings.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data shards dir (used when --selfies_csv is not given).",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default=os.environ.get("VOCAB_FILE", "checkpoints/vae_pubchem80m/vocab.json"),
        help="Path to vocab file (default: $VOCAB_FILE).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bmfm_sm_gen.api.base_tokenizer.SAFETokenizer",
        help="Tokenizer class path",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples for analysis"
    )

    args = parser.parse_args()

    global lightning_module
    lightning_module = VAELightningModule.load_from_checkpoint(
        args.checkpoint,
        map_location=torch.device("cpu"),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=args.vocab_file,
    )
    lightning_module.eval()
    device = lightning_module.device

    tokenizer = None

    latent_vectors = []
    labels = []

    real_samples = args.num_samples

    if args.selfies_csv:

        def selfies_iter():
            with open(args.selfies_csv, "r") as f:
                reader = csv.DictReader(f)
                col = args.selfies_column
                for row in reader:
                    yield row.get(col, "")

        tokenizer = getattr(lightning_module, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer not found on lightning_module; please ensure it is constructed with one."
            )

        real_latents, real_labels = encode_selfies_list(
            selfies_iter=selfies_iter(),
            tokenizer=tokenizer,
            max_len=lightning_module.model_params["max_len"],
            device=device,
            limit=real_samples,
        )
        latent_vectors.extend(real_latents)
        labels.extend(real_labels)

    else:
        data_module = MolecularDataModule(
            data_dir=args.data_dir,
            vocab_file=args.vocab_file,
            tokenizer=args.tokenizer,
            max_len=lightning_module.model_params["max_len"],
            batch_size=64,
        )
        data_module.prepare_data()
        data_module.setup()
        tokenizer = data_module.tokenizer

        dataloader = data_module.val_dataloader()
        for batch in dataloader:
            src = batch["input_ids"][:, :-1].to(device)
            with torch.no_grad():
                mu, logvar = lightning_module.model.encode(src)
                z = lightning_module.model.reparameterize(mu, logvar)
                latent_vectors.append(z.cpu().numpy())
                labels.extend([0] * z.size(0))
            if len(labels) >= real_samples:
                break

    # Stack the real latents
    latent_vectors = np.vstack(latent_vectors[:real_samples])

    # Generate molecules (SMILES), convert to SELFIES, encode, and get latents
    num_generated_samples = args.num_samples
    generated_smiles = lightning_module.generate_molecules(num_generated_samples)
    logging.info(f"Generated {len(generated_smiles)} molecules")
    valid_smiles = [s for s in generated_smiles if s is not None]
    logging.info(f"Valid generated: {len(valid_smiles)}")

    generated_latent_vectors = []
    with torch.no_grad():
        for smi in valid_smiles:
            selfies_str = smiles_to_selfies(smi)
            tok = tokenizer(
                selfies_str,
                truncation=True,
                max_length=lightning_module.model_params["max_len"],
                return_tensors="pt",
            )["input_ids"]
            src = tok[:-1].to(device)
            mu, logvar = lightning_module.model.encode(src.unsqueeze(0))
            z = lightning_module.model.reparameterize(mu, logvar)
            generated_latent_vectors.append(z.cpu().numpy())
            labels.append(1)
            if len(labels) >= args.num_samples * 2:
                break

    latent_vectors = np.concatenate(
        [np.vstack(latent_vectors), np.vstack(generated_latent_vectors)], axis=0
    )[: args.num_samples * 2]
    labels = labels[: args.num_samples * 2]

    # t-SNE + plot
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    z_embedded = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        z_embedded[:, 0],
        z_embedded[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=["Real Data", "Generated Data (Unconditional)"],
    )
    plt.savefig("latent_space.png")
    plt.show()


if __name__ == "__main__":
    main()
