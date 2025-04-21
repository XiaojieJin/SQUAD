# Original name: 20250211_annotation.py
# %%
# cellSeg_test
import time
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phenograph
import PyComplexHeatmap as pch
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from typing import Union
import sys
from pathlib import Path

# 添加 pyqupath 所在目录到 Python 路径
sys.path.append(str(Path(__file__).parent.resolve()))
from pyqupath.geojson import GeojsonProcessor
TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def update_geojson_with_cluster_csv(
    geojson_file: Union[str, Path],
    cluster_csv: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    clustering_id: str = "default_cluster"
):
    # try:
    #     from pyqupath.geojson import GeojsonProcessor
    # except ImportError:
    #     raise ImportError(
    #         "pyqupath is not installed. Please install it using `pip install git+https://github.com/wuwenrui555/pyqupath.git@v0.0.5`."
    #     )

    geojson_file = Path(geojson_file)
    geojson = GeojsonProcessor.from_path(geojson_file)

    # 读取聚类 CSV
    if isinstance(cluster_csv, (str, Path)):
        cluster_df = pd.read_csv(cluster_csv)
    else:
        cluster_df = cluster_csv.copy()

    # 建立 cellLabel → cluster 映射字典
    name_dict = dict(zip(cluster_df["cellLabel"].astype(str), cluster_df["cluster"]))

    # 筛选 geojson 中的细胞并更新 classification 字段
    geojson.gdf = geojson.gdf[geojson.gdf["name"].isin(name_dict.keys())]
    geojson.update_classification(name_dict)

    # 输出文件路径
    output_file = Path(output_dir)

    # 保存
    geojson.output_geojson(output_file)
    print(f"✅ GeoJSON with cluster saved to: {output_file}")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Annotation")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--annotation_results', type=str, default='annotation_results', help='Annotation results directory')
    parser.add_argument('--segmentation_results', type=str, default='segmentation_results_0', help='Segmentation results directory')
    parser.add_argument('--preprocessing_results', type=str, default='preprocessing_results', help='Preprocessing results directory')
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
output_dir = Path(args.data_dir) / args.annotation_results
output_dir.mkdir(parents=True, exist_ok=True)
input_dir = Path(args.data_dir) / args.preprocessing_results
segmentation_dir = Path(args.data_dir) / args.segmentation_results
# 设置标记物
MARKERS_PHENOTYPIC = args.markers_phenotypic

def phenograph_cluster(
    data: pd.DataFrame,
    markers: list[str],
    output_dir: str,
    tag: str = None,
    k: int = 50,
    seed: int = 123,
):
    """
    Perform PhenoGraph clustering on the data

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing marker expression data
    markers : list[str]
        List of markers to use for clustering
    output_dir : str
        Output directory
    tag : str
        Tag for the output directory (default=None)
    k : int
        Number of nearest neighbors (default=30)
    seed : int
        Random seed for reproducibility (default=123)

    Returns:
    --------
    None
    """
    if tag is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path(output_dir) / tag
    root_dir.mkdir(parents=True, exist_ok=True)

    data_cluster = data[markers]
    data_desc = data_cluster.describe()
    if np.sum(data_desc.loc["min"] != 0) + np.sum(data_desc.loc["max"] != 1) > 0:
        raise ValueError("Data not normalized")

    with open(root_dir / "markers.txt", "w") as f:
        f.write("\n".join(markers))

    # Run PhenoGraph
    communities, graph, Q = phenograph.cluster(data_cluster.values, k=k, seed=seed)

    data["cluster"] = communities
    data.to_csv(root_dir / "cluster.csv", index=False)

def plot_cluster_heatmap(data, metadata, markers, figsize=(15, 10)):
    """Plot heatmap of marker expression for each cluster."""
    # 准备数据
    data_cluster = data[["cluster"] + markers]
    cluster_means = data_cluster.groupby("cluster").mean()
    
    # 创建热图
    plt.figure(figsize=figsize)
    sns.heatmap(cluster_means, cmap="viridis", center=0, vmin=-2, vmax=2)
    plt.title("Marker Expression by Cluster")
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("data/annotation_results")
    plt.savefig(output_dir / "cluster_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()

def run_cluster_step_01():
    tag = "step_01"
    input_f = input_dir / "combined_dataScaleSize_rm_extreme_normDAPI_arcsinh_quantile.csv"

    data = pd.read_csv(input_f)
    # markers =[
    #     "CD163",
    #     "CD68",
    #     "CD31",
    #     "CD11c",
    #     "Pax5",
    #     "CD20",
    #     "CD8",
    #     "FoxP3",
    #     "CD3"
    # ]
    # markers = MARKERS_PHENOTYPIC
    markers = MARKERS_PHENOTYPIC
    phenograph_cluster(data, markers, str(output_dir), tag)
    plot_cluster_heatmap(data, pd.DataFrame({"id": ["sample1"], "tissue_type": ["tumor"]}), markers)
    geojson_file = segmentation_dir / "segmentation_mask.geojson"
    cluster_csv = output_dir / "step_01" / "cluster.csv"
    output_dir_0 = output_dir / "step_01" / "segmentation_mask_updated.geojson"
    update_geojson_with_cluster_csv(
        geojson_file=geojson_file,
        cluster_csv=cluster_csv,
        output_dir=output_dir_0
    )


def run_cluster_step_02_immune():
    output_dir = Path("data/annotation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "step_02_immune"
    input_f = Path("data/preprocessing_results/combined_dataScaleSize_rm_extreme_normDAPI_arcsinh_quantile.csv")

    data = pd.read_csv(input_f)
    markers = [
        "CD3",
        "CD4",
        "CD8",
        "FoxP3",
        "TCF1",
        "GZMB-250",
        "Tox1-2",
        "PD1",
        "LAG3-250",
        "TIM3",
        "TIGIT"
    ]
    phenograph_cluster(data, markers, str(output_dir), tag)
    plot_cluster_heatmap(data, pd.DataFrame({"id": ["sample1"], "tissue_type": ["tumor"]}), markers)


def run_cluster_step_02_nonimmune():
    output_dir = Path("data/annotation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "step_02_nonimmune"
    input_f = Path("data/preprocessing_results/combined_dataScaleSize_rm_extreme_normDAPI_arcsinh_quantile.csv")

    data = pd.read_csv(input_f)
    markers = [
        "CD20",
        "CD163",
        "CD68",
        "CD11c",
        "CD31",
        "ATP5A",
        "IDO1",
        "PDL1-250",
        "HLA1",
        "Ki67",
        "BCL2",
        "Pax5"
    ]
    phenograph_cluster(data, markers, str(output_dir), tag)
    plot_cluster_heatmap(data, pd.DataFrame({"id": ["sample1"], "tissue_type": ["tumor"]}), markers)


# %%
if __name__ == "__main__":
    run_cluster_step_01()
    # run_cluster_step_02_immune()
    # run_cluster_step_02_nonimmune()
    None
