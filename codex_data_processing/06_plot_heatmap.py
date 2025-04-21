import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import argparse

# Define phenotypic markers
# MARKERS_PHENOTYPIC = [
#     "CD3",
#     "CD4",
#     "CD8",
#     "FoxP3",
#     "CD20",
#     "CD163",
#     "CD68",
#     "CD11c",
#     "CD31",
#     "PDL1-250",
#     "HLA1",
#     "Ki67",
#     "BCL2",
#     "Pax5"
# ]

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Plot Heatmap")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--annotation_results', type=str, default='annotation_results', help='Annotation results directory')
    parser.add_argument('--markers_phenotypic', nargs='+', default=[
        "1_CD3",
        "1_CD4-150ms",
        "1_FoxP3",
        "1_CD8",
        "1_CD68",
        "3_CD206-250ms",
        "1_CD163",
        "1_CD11c"
    ], help='List of phenotypic markers')
    return parser.parse_args()

args = parse_args()

# 设置路径
data_dir = Path(args.data_dir)
input_f = data_dir / args.annotation_results / "step_01/cluster.csv"
output_f = data_dir / args.annotation_results / "cluster_heatmap_phenotypic_zscore.png"

# 设置标记物
MARKERS_PHENOTYPIC = args.markers_phenotypic

# Read clustering results
data = pd.read_csv(input_f)

# Calculate mean expression for each marker in each cluster
cluster_means = data.groupby('cluster')[MARKERS_PHENOTYPIC].mean()

# Calculate z-score for each marker
scaler = StandardScaler()
zscore_data = pd.DataFrame(
    scaler.fit_transform(cluster_means),
    index=cluster_means.index,
    columns=cluster_means.columns
)

# Set plot style
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.5

# Create figure
fig = plt.figure(figsize=(12, 10))

# Set height ratio for subplots
gs = plt.GridSpec(2, 1, height_ratios=[1, 4])

# Create top bar plot
ax1 = fig.add_subplot(gs[0])
cluster_sizes = data['cluster'].value_counts().sort_index()
ax1.bar(range(len(cluster_sizes)), cluster_sizes.values, color='gray', edgecolor='black')
ax1.set_ylabel('Cell Count')
ax1.set_xticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Create bottom heatmap with z-score
ax2 = fig.add_subplot(gs[1])
sns.heatmap(zscore_data.T, 
            cmap='RdBu_r',  # 使用红蓝配色方案
            center=0,       # 将0值设置为白色
            vmin=-3,        # 设置z-score范围
            vmax=3,
            cbar_kws={'label': 'Z-score'},
            ax=ax2,
            xticklabels=True,
            yticklabels=True)

# Set labels and title
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Markers')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(output_f, 
            dpi=300, 
            bbox_inches='tight')
plt.close()

print(f"Z-score heatmap saved to: {output_f}") 