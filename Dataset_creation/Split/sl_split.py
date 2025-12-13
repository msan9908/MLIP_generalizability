import os
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib.pyplot as plt
from sparse import load_npz

def load_avg_soaps(soaps_path, refcodes_path):
    refcodes = np.genfromtxt(refcodes_path, delimiter=',', dtype=str).tolist()
    avg_soaps = []
    num_atoms_list = []  # New: Store number of atoms per MOF
    
    for refcode in refcodes:
        soap_file = os.path.join(soaps_path, f'soap_{refcode}.npz')
        soap_matrix = load_npz(soap_file).todense()
        num_atoms = soap_matrix.shape[0]  # Number of rows = number of atoms
        num_atoms_list.append(num_atoms)
        avg_soaps.append(soap_matrix.mean(axis=0))
    
    return np.vstack(avg_soaps), refcodes, np.array(num_atoms_list)  # Now returns 3 values

if __name__ == "__main__":
    # Load data - now gets num_atoms_list too
    basepath = os.getcwd()
    soaps_path = os.path.join(basepath, 'soap_matrices')
    refcodes_path = os.path.join('..', 'qmof-refcodes.csv')
    avg_soaps, refcodes, num_atoms = load_avg_soaps(soaps_path, refcodes_path)  # Modified
    
    # Create size masks
    small_mofs_mask = num_atoms < 100
    large_mofs_mask = num_atoms >= 100
    
    # Split data
    
    
    # Plotting (modified from previous version)
    umap = UMAP(random_state=42)
    X_2d = umap.fit_transform(avg_soaps)
    
    # Enhanced training plot
    plt.figure(figsize=(10, 8))
    # Small training MOFs
    train_small = np.where(small_mofs_mask)[0]
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')

    plt.scatter(X_2d[train_small, 0], X_2d[train_small, 1],
                c='#1f77b4', alpha=0.7, s=40,  label='Small MOFs (<100 atoms)')
    plt.title('MOF Size Distribution', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend()
    plt.savefig('train_size_distribution.png', dpi=300)

    plt.close()
    
    # Plot 2: Test set
    plt.figure(figsize=(10, 8))
    # Large training MOFs
    train_large = np.where(large_mofs_mask)[0]
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='All')

    plt.scatter(X_2d[train_large, 0], X_2d[train_large, 1],
                c='#d62728', alpha=0.7, s=40, label='Large MOFs (≥100 atoms)')
    
    plt.title('MOF Size Distribution', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_size_distribution.png', dpi=300)
    plt.close()
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap('Pastel2_r')
        # Plot 3: Cluster overview (custom style)
    
    # Pick two colors for groups of clusters
    colors = [cmap(i / 2) for i in range(3)] 

    
    
    plt.scatter(X_2d[train_small, 0], X_2d[train_small, 1],
                color=colors[0], s=10, alpha=0.7, label="Small group")
    plt.scatter(X_2d[train_large, 0], X_2d[train_large, 1],
                color=colors[1], s=10, alpha=0.7, label="Large group")
    
        
        # Highlight one point in the cluster
        

    # Clean up plot
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.savefig("sl_highlight_n.png", dpi=900, bbox_inches="tight")
    plt.close()