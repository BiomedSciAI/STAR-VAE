import argparse

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule


def main():
    parser = argparse.ArgumentParser(description="Conditional Latent Space Analysis")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_labels", type=int, default=10, help="Number of label bins"
    )
    parser.add_argument(
        "--samples_per_label", type=int, default=50, help="Samples per label"
    )
    parser.add_argument(
        "--use_conditional_prior",
        action="store_true",
        help="Whether to use the conditional prior p(z|y)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="conditional_latent_space.png",
        help="Output image file",
    )
    args = parser.parse_args()

    model = CVAELightningModule.load_from_checkpoint(
        args.checkpoint,
        map_location=torch.device("cpu"),
    )
    model.eval()

    latent_vectors = []
    label_values = []

    bin_edges = np.linspace(1.0, args.num_labels, args.num_labels + 1)

    for i in range(len(bin_edges) - 1):
        y_val = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        labels = (
            torch.rand(args.samples_per_label, 1) * (bin_edges[i + 1] - bin_edges[i])
            + bin_edges[i]
        )

        if args.use_conditional_prior:
            mu, logvar = model.model.get_conditional_prior(labels)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = torch.randn(args.samples_per_label, model.model.latent_dim)

        zc = model.model.modulation(z, labels)
        latent_vectors.append(zc.detach())
        label_values.extend(
            [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"] * args.samples_per_label
        )

    latent_vectors = torch.vstack(latent_vectors).numpy()
    label_values = np.array(label_values)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
    z_embedded = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(16, 12))
    unique_labels = sorted(set(label_values))
    colors = cm.get_cmap("tab20", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        indices = label_values == lbl
        plt.scatter(
            z_embedded[indices, 0],
            z_embedded[indices, 1],
            label=f"Label {lbl}",
            alpha=0.4,
            color=colors(i),
        )

    plt.legend(title="Label Bins")
    prior_type = "Conditional Prior" if args.use_conditional_prior else "Standard Prior"
    plt.title(f"Conditional Latent Space (t-SNE) - {prior_type}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
