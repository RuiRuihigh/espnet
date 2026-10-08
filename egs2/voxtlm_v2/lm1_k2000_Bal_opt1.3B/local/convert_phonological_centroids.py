"""Wrap the fixed Phonological Tokenizer codebook (centroids.npy) into the
sklearn-KMeans-shaped pickle expected by scripts/feats/perform_kmeans.sh and
pyscripts/feats/dump_km_label.py, so the existing kmeans-labeling pipeline
can assign discrete units by nearest-centroid lookup without training a
kmeans model of our own.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--centroids_path",
        type=str,
        required=True,
        help="Path to the Phonological Tokenizer centroids.npy",
    )
    parser.add_argument(
        "--km_path",
        type=str,
        required=True,
        help="Output path for the km model pickle "
        "(e.g. ${km_dir}/km_${nclusters}.mdl)",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=2000,
        help="Expected number of centroids, for a sanity check",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite km_path even if it already exists",
    )
    return parser


def main(args):
    km_path = Path(args.km_path)
    if km_path.exists() and not args.force:
        print(f"{km_path} already exists. Skip conversion.")
        return

    centroids = np.load(args.centroids_path)
    assert centroids.ndim == 2, f"Expected a 2D array, got shape {centroids.shape}"
    assert centroids.shape[0] == args.n_clusters, (
        f"Expected {args.n_clusters} centroids, got {centroids.shape[0]} "
        f"from {args.centroids_path}"
    )

    km_model = MiniBatchKMeans(n_clusters=args.n_clusters)
    km_model.cluster_centers_ = centroids.astype(np.float32)

    km_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(km_model, km_path)
    print(f"Wrote {args.n_clusters}-cluster km model to {km_path}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
