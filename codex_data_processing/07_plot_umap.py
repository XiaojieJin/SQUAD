import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import umap
from sklearn.preprocessing import StandardScaler
import argparse

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Plot UMAP")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--annotation_results', type=str, default='annotation_results', help='Annotation results directory')
    parser.add_argument('--markers_phenotypic', nargs='+', default=[
        "1_CD3",
        "1_CD4-150ms",
        "1_FoxP3",# low-quality marker
        "1_CD8",
        "1_CD68",
        "3_CD206-250ms",# low-quality marker
        "1_CD163",
        "1_CD11c"
    ], help='List of phenotypic markers')
    return parser.parse_args()

args = parse_args()

# 设置路径
data_dir = Path(args.data_dir)
input_f = data_dir / args.annotation_results / "step_01/cluster.csv"
output_dir = data_dir / args.annotation_results / "umap"
output_dir.mkdir(exist_ok=True, parents=True)

# 设置标记物
MARKERS_PHENOTYPIC = args.markers_phenotypic

# Read clustering results
data = pd.read_csv(input_f)

# Prepare data for UMAP
X = data[MARKERS_PHENOTYPIC].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run UMAP
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding = reducer.fit_transform(X_scaled)

# Create UMAP visualization
plt.figure(figsize=(10, 8))

# Plot UMAP with cluster colors
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=data['cluster'],
    cmap='tab20',
    alpha=0.6,
    s=1
)

# Add legend
legend1 = plt.legend(*scatter.legend_elements(),
                    loc="upper right",
                    title="Clusters",
                    bbox_to_anchor=(1.15, 1))

# Set labels and title
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP of Phenotypic Markers')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save figure
plt.savefig(output_dir / "umap_clusters.png",
            dpi=300,
            bbox_inches='tight')
plt.close()

# Create UMAP with marker expression
for marker in MARKERS_PHENOTYPIC:
    plt.figure(figsize=(10, 8))
    
    # Plot UMAP with marker expression
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=data[marker],
        cmap='viridis',
        alpha=0.6,
        s=1
    )
    
    # Add colorbar
    plt.colorbar(scatter, label='Expression')
    
    # Set labels and title
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'UMAP - {marker} Expression')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / f"umap_{marker}.png",
                dpi=300,
                bbox_inches='tight')
    plt.close()

print(f"UMAP visualizations saved to: {output_dir}") 