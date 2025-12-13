import os
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sparse import load_npz

def load_avg_soaps(soaps_path, refcodes_path):
    refcodes = np.genfromtxt(refcodes_path, delimiter=',', dtype=str).tolist()
    avg_soaps = []
    for refcode in refcodes:
        soap_file = os.path.join(soaps_path, f'soap_{refcode}.npz')
        soap_matrix = load_npz(soap_file).todense()
        avg_soaps.append(soap_matrix.mean(axis=0))
    return np.vstack(avg_soaps), refcodes

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
    # Load data
    basepath = os.getcwd()
    soaps_path = os.path.join(basepath, 'soap_matrices')
    refcodes_path = os.path.join('..', 'qmof-refcodes.csv')
    avg_soaps, refcodes = load_avg_soaps(soaps_path, refcodes_path)

    # Random split
    train_indices, test_indices = random_split(avg_soaps, train_ratio=0.5, seed=3)

    # Save indices and refcodes
    train_refcodes = np.array(refcodes)[train_indices].tolist()
    test_refcodes = np.array(refcodes)[test_indices].tolist()
    
    np.savetxt('train_indices_random_3.txt', train_indices, fmt='%d')
    np.savetxt('test_indices_random_3.txt', test_indices, fmt='%d')
    np.savetxt('train_refcodes_random_3.csv', train_refcodes, fmt='%s')
    np.savetxt('test_refcodes_random_3.csv', test_refcodes, fmt='%s')
    
    # UMAP embedding
    umap = UMAP(random_state=42)
    X_2d = umap.fit_transform(avg_soaps)

    # Plot 1: Train set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')
    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1], 
                c='#1f77b4', alpha=0.7, s=40, label='Train (Random)')
    plt.title('Random Split: Training Set', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend()
    plt.savefig('random_train_3.png', dpi=300)
    plt.close()

    # Plot 2: Test set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1], 
                c='#d62728', alpha=0.7, s=40, label='Test (Random)')
    plt.title('Random Split: Test Set', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend()
    plt.savefig('random_test_3.png', dpi=300)
    plt.close()

    # Plot 3: Overview
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap('Pastel2_r')
    colors = [cmap(i / 2) for i in range(3)]

    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1],
                color=colors[0], s=20, alpha=1, label="Train group")
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1],
                color=colors[1], s=20, alpha=0.7, label="Test group")

    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("random_highlight_n.png", dpi=900, bbox_inches="tight")
    plt.close()
