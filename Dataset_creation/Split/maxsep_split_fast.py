import os
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sparse import load_npz

def load_avg_soaps(soaps_path):
    """
    Load OMOL SOAP matrices, compute average per MOF, and number of atoms.
    """
    soap_files = sorted([f for f in os.listdir(soaps_path) if f.endswith(".npz")])
    avg_soaps = []
    refcodes = []
    num_atoms_list = []

    for f in soap_files:
        full_path = os.path.join(soaps_path, f)
        if os.path.getsize(full_path) < 100:  # skip tiny/corrupt files
            print("Skipping tiny file:", f)
            continue
        try:
            mat = load_npz(full_path).todense()
            avg_soaps.append(mat.mean(axis=0))  # average over atoms
            num_atoms_list.append(mat.shape[0])
            refcodes.append(f.replace("soap_", "").replace(".npz", ""))
        except Exception as e:
            print("Failed to load", f, ":", e)
            continue

    if len(avg_soaps) == 0:
        raise RuntimeError(f"No valid SOAP files loaded from: {soaps_path}")

    return np.vstack(avg_soaps), refcodes, np.array(num_atoms_list)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; returns float32 to reduce memory."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return (vectors / norms).astype(np.float32, copy=False)


def max_separation_split_subset_on_the_fly(
    vectors: np.ndarray,
    train_ratio: float = 0.5,
    seed: int = 3,
    train_subset_size: int = 1000,
    cand_subset_size: int = 1000,
    verbose: bool = True,
):
    """
    Approximate 'max separation' split without building an NxN similarity matrix.
    At each step:
      - sample a candidate subset from remaining points
      - sample a train subset from the (fixed) seed train indices (to mirror your original loop)
      - pick the least similar candidate (cosine) to the sampled train subset
    IMPORTANT: This mirrors your original code's behavior where 'train_indices'
               inside the loop is NOT expanded with chosen test points.
               The only difference is that we compute similarity on random subsets.
    """
    rng = np.random.default_rng(seed)
    N = vectors.shape[0]
    desired_test_size = N - int(N * train_ratio)
    if desired_test_size <= 0:
        # Nothing to move to test; all training
        all_idx = np.arange(N)
        return all_idx, np.array([], dtype=int)

    # Normalize once
    norm_vecs = normalize_vectors(vectors)

    # Initial seed (same as original)
    all_indices = np.arange(N)
    seed_idx = int(rng.integers(0, N))
    train_indices_seed = np.array([seed_idx], dtype=int)

    # Remaining pool (candidates for test)
    remaining_indices = np.setdiff1d(all_indices, train_indices_seed, assume_unique=False)

    test_indices = []

    for step in range(desired_test_size):
        # Candidate subset from remaining
        if remaining_indices.size > cand_subset_size:
            cand_subset = rng.choice(remaining_indices, size=cand_subset_size, replace=False)
        else:
            cand_subset = remaining_indices

        # Train subset from the fixed seed set (mirrors original behavior)
        # (In your original code, this set stayed constant through the loop.)
        if train_indices_seed.size > train_subset_size:
            train_subset = rng.choice(train_indices_seed, size=train_subset_size, replace=False)
        else:
            train_subset = train_indices_seed

        # Cosine sims only on chosen subsets
        # shape: (len(cand_subset), len(train_subset))
        sims = norm_vecs[cand_subset] @ norm_vecs[train_subset].T
        # Reduce across train subset to the "best similarity" for each candidate
        best_sims = sims.max(axis=1)

        # Pick the least similar candidate in the subset
        least_sim_local = int(np.argmin(best_sims))
        least_sim_global_idx = int(cand_subset[least_sim_local])

        # Move to test, remove from remaining
        test_indices.append(least_sim_global_idx)
        remaining_indices = remaining_indices[remaining_indices != least_sim_global_idx]

        # Progress
        if verbose and (step + 1) % max(1, desired_test_size // 20) == 0:
            pct = 100 * (step + 1) / desired_test_size
            print(f"[Subset MaxSep] Step {step+1}/{desired_test_size} ({pct:.1f}%)")

        # Early stop if nothing remains
        if remaining_indices.size == 0:
            break

    # Final train set: whatever remained plus the initial seed (same as your original)
    train_indices = np.concatenate([remaining_indices, train_indices_seed]).astype(int, copy=False)
    return train_indices, np.array(test_indices, dtype=int)

def random_split(avg_soaps, train_ratio=0.5, seed=42):
    np.random.seed(seed)
    n_samples = avg_soaps.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_train = int(train_ratio * n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return train_indices, test_indices


if __name__ == "__main__":
    basepath = os.getcwd()
    soaps_path = os.path.join(basepath, "soap_matrices")

    seed = 2
    train_ratio = 0.5
    subset_size = 3000  # number of training points to sample at each step

    # ---- Load data ----
    avg_soaps, refcodes, num_atoms = load_avg_soaps(soaps_path)
    print(f"Loaded {len(avg_soaps)} SOAP descriptors")

    # ---- Size masks ----
    small_mask = num_atoms < 150
    large_mask = num_atoms >= 150

    # Perform approximate MaxSep
    #train_indices, test_indices = max_separation_split_subset_on_the_fly(  avg_soaps, train_ratio, seed, subset_size,subset_size)

    # Save indices and refcodes
    #np.savetxt(f"train_indices_maxsep_omol_{seed}.txt", train_indices, fmt="%d")
    #np.savetxt(f"test_indices_maxsep_omol_{seed}.txt", test_indices, fmt="%d")
    #np.savetxt(f"train_refcodes_maxsep_omol_{seed}.csv", np.array(refcodes)[train_indices], fmt="%s")
    #np.savetxt(f"test_refcodes_maxsep_omol_{seed}.csv", np.array(refcodes)[test_indices], fmt="%s")



    train_indices = np.loadtxt(f"train_indices_maxsep_omol_{seed}.txt", dtype=int)
    test_indices = np.loadtxt(f"test_indices_maxsep_omol_{seed}.txt", dtype=int)
    # Compute UMAP
    umap = UMAP(random_state=42)
    X_2d = umap.fit_transform(avg_soaps)

    # Plot training set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="lightgray", alpha=0.3, s=10, label="All")
    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1],
                c="#1f77b4", alpha=0.7, s=40, label="Train (MaxSep)")
    plt.title(f"Maximal Separation Split (Seed={seed}): Training Set", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.legend()
    plt.savefig(f"maxsep_train_omol_{seed}.png", dpi=300)
    plt.close()

    # Plot test set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="lightgray", alpha=0.3, s=10, label="All")
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1],
                c="#d62728", alpha=0.7, s=40, label="Test (MaxSep)")
    plt.title(f"Maximal Separation Split (Seed={seed}): Test Set", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.legend()
    plt.savefig(f"maxsep_test_omol_{seed}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))

    cmap = plt.cm.get_cmap('Pastel2_r')
        # Plot 3: Cluster overview (custom style)
    
    # Pick two colors for groups of clusters
    colors = [cmap(i / 2) for i in range(3)] 

    
    # ⚠️ Instead of recomputing kmeans, we reuse train/test split
    # Here: treat "train" as cluster group A, "test" as cluster group B
    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1],
                color=colors[0], s=30, alpha=0.3, label="Train group")
    
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1],
                color=colors[1], s=30, alpha=0.7, label="Test group")
    
        
        # Highlight one point in the cluster
        

    # Clean up plot
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("maxsep_highlight_n_omol.png", dpi=900, bbox_inches="tight")
    plt.close()

    train_indices, test_indices = random_split(avg_soaps, train_ratio=0.5, seed=3)


    plt.figure(figsize=(6, 6))

    cmap = plt.cm.get_cmap('Pastel2_r')
        # Plot 3: Cluster overview (custom style)
    
    # Pick two colors for groups of clusters
    colors = [cmap(i / 2) for i in range(3)] 

    
    # ⚠️ Instead of recomputing kmeans, we reuse train/test split
    # Here: treat "train" as cluster group A, "test" as cluster group B
    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1],
                color=colors[0], s=30, alpha=0.3, label="Train group")
    
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1],
                color=colors[1], s=30, alpha=0.7, label="Test group")
    
        
        # Highlight one point in the cluster
        

    # Clean up plot
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("random_highlight_omol.png", dpi=900, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap('Pastel2_r')
        # Plot 3: Cluster overview (custom style)
    
    # Pick two colors for groups of clusters
    colors = [cmap(i / 2) for i in range(3)] 

    train_small = np.where(small_mask)[0]
    train_large = np.where(large_mask)[0]
    
    plt.scatter(X_2d[train_small, 0], X_2d[train_small, 1],
                color=colors[0], s=30, alpha=0.3,  label="Small group")
    plt.scatter(X_2d[train_large, 0], X_2d[train_large, 1],
                color=colors[1], s=30, alpha=0.7,  label="Large group")
    
        
        # Highlight one point in the cluster
        

    # Clean up plot
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("sl_highlight_n_omol.png", dpi=900, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))

    cmap = plt.cm.get_cmap('Pastel2_r')
        # Plot 3: Cluster overview (custom style)
    
    # Pick two colors for groups of clusters
    colors = [cmap(i / 2) for i in range(3)] 
            # Highlight one point in the cluster
    train_indices = np.loadtxt(f"train_indices_cluster_omol_{seed}.txt", dtype=int)
    test_indices = np.loadtxt(f"test_indices_cluster_omol_{seed}.txt", dtype=int)

    
    # ⚠️ Instead of recomputing kmeans, we reuse train/test split
    # Here: treat "train" as cluster group A, "test" as cluster group B
    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1],
                color=colors[0], s=30, alpha=0.3, label="Train group")
    
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1],
                color=colors[1], s=30, alpha=0.7, label="Test group")
    
        

        

    # Clean up plot
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("cluster_highlight_n_omol.png", dpi=900, bbox_inches="tight")
    plt.close()

