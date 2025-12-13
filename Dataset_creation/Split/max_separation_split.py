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

def compute_cosine_similarity(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = vectors / norms
    return normalized @ normalized.T

def max_separation_split(similarity_matrix, train_ratio=0.5, seed=3):#original seed = 42 v2 seed=2 v3 seed=3
    np.random.seed(seed)
    N = similarity_matrix.shape[0]
    desired_test_size = N - int(N * train_ratio)
    all_indices = np.arange(N)
    train_indices = [np.random.choice(N)]
    remaining_indices = np.setdiff1d(all_indices, train_indices).tolist()
    test_indices = []
    for _ in range(desired_test_size):
        sims = similarity_matrix[remaining_indices][:, train_indices].max(axis=1)
        least_similar_idx = remaining_indices[np.argmin(sims)]
        test_indices.append(least_similar_idx)
        remaining_indices.remove(least_similar_idx)
    train_indices = remaining_indices + train_indices
    return np.array(train_indices), np.array(test_indices)


if __name__ == "__main__":
    # Load data and split
    basepath = os.getcwd()
    soaps_path = os.path.join(basepath, 'soap_matrices')
    refcodes_path = os.path.join('..', 'qmof-refcodes.csv')
    avg_soaps, refcodes = load_avg_soaps(soaps_path, refcodes_path)
    similarity_matrix = compute_cosine_similarity(avg_soaps)
    train_indices, test_indices = max_separation_split(similarity_matrix)
    
    # Save indices and refcodes
    train_refcodes = np.array(refcodes)[train_indices].tolist()
    test_refcodes = np.array(refcodes)[test_indices].tolist()
    
    np.savetxt('train_indices_maxsep_3.txt', train_indices, fmt='%d')
    np.savetxt('test_indices_maxsep_3.txt', test_indices, fmt='%d')
    np.savetxt('train_refcodes_maxsep_3.csv', train_refcodes, fmt='%s')
    np.savetxt('test_refcodes_maxsep_3.csv', test_refcodes, fmt='%s')
    
    # Compute UMAP
    umap = UMAP(random_state=42)
    X_2d = umap.fit_transform(avg_soaps)
    
    # Plot 1: Train set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')
    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1], 
                c='#1f77b4', alpha=0.7, s=40, label='Train (MaxSep)')
    plt.title('Maximal Separation Split: Training Set', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend()
    plt.savefig('maxsep_train_3.png', dpi=300)
    plt.close()
    
    # Plot 2: Test set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1], 
                c='#d62728', alpha=0.7, s=40, label='Test (MaxSep)')
    plt.title('Maximal Separation Split: Test Set', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.legend()
    plt.savefig('maxsep_test_3.png', dpi=300)
    plt.close()
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap('Pastel2_r')
        # Plot 3: Cluster overview (custom style)
    
    # Pick two colors for groups of clusters
    colors = [cmap(i / 2) for i in range(3)] 

    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1],
                color=colors[0], s=30, alpha=0.3, label="Train group")
    
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1],
                color=colors[1], s=30, alpha=0.7, label="Test group")
    
        
        # Highlight one point in the cluster
        

    # Clean up plot
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("maxsep_highlight_n.png", dpi=900, bbox_inches="tight")
    plt.close()

    cmap = plt.cm.get_cmap('Pastel2_r')
        # Plot 3: Cluster overview (custom style)
    
    # Pick two colors for groups of clusters
    colors = [cmap(i / 2) for i in range(3)] 

    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1],
                color=colors[0], s=30, alpha=0.3, label="Train group")
    

    
        
        # Highlight one point in the cluster
        

    # Clean up plot
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("train only_n.png", dpi=900, bbox_inches="tight")
    plt.close()
    