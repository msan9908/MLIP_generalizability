import os
import numpy as np
from sklearn.cluster import KMeans
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

def cluster_split(avg_soaps, train_ratio=0.5, n_clusters=20, seed=42):#original seed = 42 v2 seed=2 v3 seed=3
    np.random.seed(seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(avg_soaps)
    unique_clusters = np.unique(cluster_labels)
    n_train_clusters = int(train_ratio * len(unique_clusters))
    train_clusters = np.random.choice(unique_clusters, n_train_clusters, replace=False)
    test_clusters = np.setdiff1d(unique_clusters, train_clusters)
    train_indices = np.where(np.isin(cluster_labels, train_clusters))[0]
    test_indices = np.where(np.isin(cluster_labels, test_clusters))[0]
    return train_indices, test_indices

if __name__ == "__main__":
    # Load data and split
    basepath = os.getcwd() # Get current working directory adjust if needed
    soaps_path = os.path.join(basepath, 'soap_matrices')
    refcodes_path = os.path.join('..', 'qmof-refcodes.csv')
    avg_soaps, refcodes = load_avg_soaps(soaps_path, refcodes_path)
    train_indices, test_indices = cluster_split(avg_soaps, train_ratio=0.5, n_clusters=20, seed=3)#original seed = 42 v2 seed=2 v3 seed=3
    
    # Save indices and refcodes
    train_refcodes = np.array(refcodes)[train_indices].tolist()
    test_refcodes = np.array(refcodes)[test_indices].tolist()
    
    np.savetxt('train_indices_cluster_3.txt', train_indices, fmt='%d')
    np.savetxt('test_indices_cluster_3.txt', test_indices, fmt='%d')
    np.savetxt('train_refcodes_cluster_3.csv', train_refcodes, fmt='%s')
    np.savetxt('test_refcodes_cluster_3.csv', test_refcodes, fmt='%s')
    
    # Compute UMAP
    umap = UMAP(random_state=42)
    X_2d = umap.fit_transform(avg_soaps)
    
    # Plot 1: Train set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')
    plt.scatter(X_2d[train_indices, 0], X_2d[train_indices, 1], 
                c='#1f77b4', alpha=0.7, s=40, label='Train (Cluster)')
    plt.title('Cluster-Based Split: Training Set', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend()
    plt.savefig('cluster_train_3.png', dpi=300)
    plt.close()
    
    # Plot 2: Test set
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')
    plt.scatter(X_2d[test_indices, 0], X_2d[test_indices, 1], 
                c='#d62728', alpha=0.7, s=40, label='Test (Cluster)')
    plt.title('Cluster-Based Split: Test Set', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.legend()
    plt.savefig('cluster_test_3.png', dpi=300)
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
    plt.savefig("cluster_highlight_n.png", dpi=900, bbox_inches="tight")
    plt.close()

