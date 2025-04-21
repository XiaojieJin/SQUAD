# Original name: 20250211_preprocessing.py
# %%
# cellSeg_test
import contextlib
import os
import re
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from pyqupath.ometiff import load_tiff_to_dict
from skimage import filters
from tqdm import tqdm

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--segmentation_results', type=str, default='segmentation_results_0', help='Segmentation results directory')
    parser.add_argument('--preprocessing_results', type=str, default='preprocessing_results', help='Preprocessing results directory')
    parser.add_argument('--markers', nargs='+', default=[
        "1_DAPI",
        "1_CD3",
        "1_CD4-150ms",
        "1_FoxP3",
        "1_CD8",
        "1_CD68",
        "3_CD206-250ms",
        "1_CD163",
        "1_CD11c"
    ], help='List of markers')
    return parser.parse_args()

args = parse_args()

# 设置路径
ometiff_dir = Path(args.data_dir)
output_dir = Path(args.data_dir) / args.preprocessing_results
output_dir.mkdir(exist_ok=True, parents=True)
segmentation_dir = ometiff_dir / args.segmentation_results

# 设置标记物
MARKER_NAMES = args.markers

@contextlib.contextmanager
def is_verbose(verbose: bool):
    """
    Context manager that controls output visibility based on verbosity.

    This manager either allows normal output/error streams or redirects them
    to the null device, depending on the verbose flag.

    Parameters
    ----------
    verbose : bool
        If True, preserves all standard output and error streams.
        If False, suppresses all output by redirecting to os.devnull.

    Yields
    ------
    None
        This context manager doesn't return a value but controls I/O streams.

    Examples
    --------
    >>> with is_verbose(verbose=True):
    ...     print("This message will be visible")
    This message will be visible

    >>> with is_verbose(verbose=False):
    ...     print("This message will be suppressed")

    Notes
    -----
    Uses contextlib.redirect_stdout and redirect_stderr internally when
    suppressing output. Thread-safe and exception-safe through contextlib
    resource management.
    """
    if verbose:
        # Bypass redirection when verbose is True
        yield
    else:
        # Redirect both streams to null device when verbose is False
        with open(os.devnull, "w") as devnull:
            with (
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                yield

# If gating, it is not needed
def filter_cell_by_dapi_otsu():
    """
    Filter cells by DAPI Otsu threshold.
    """

    ometiff_f = ometiff_dir / "combined_images_gated.ome.tiff"
    segmentation_f = segmentation_dir / "segmentation_mask.tiff"
    
    with is_verbose(False):
        img_dapi = load_tiff_to_dict(ometiff_f, "ome.tiff", ["1_DAPI"])["1_DAPI"]
    
    segmentation_mask = tifffile.imread(segmentation_f)
    
    threshold = filters.threshold_otsu(img_dapi)
    otsu_mask = img_dapi > threshold
    
    cell_labels = np.unique(segmentation_mask)
    cell_labels = cell_labels[cell_labels != 0].tolist()
    cell_labels_above = np.unique(segmentation_mask * otsu_mask)
    
    reg_df = pd.DataFrame({
        "id": "sample1",
        "cell_labels": cell_labels,
        "dapi_otsu_threshold": threshold,
        "cell_labels_above": [x in cell_labels_above for x in cell_labels],
    })
    
    output_f = output_dir / "otsu_threshold.csv"
    reg_df.to_csv(str(output_f), index=False)

def combine_data_scale_size():
    """
    Bind multiple dataScaleSize.csv files into one.
    """
    data_dir = segmentation_dir
    
    data_f = data_dir / "dataScaleSize.csv"
    data_df = pd.read_csv(data_f)
    data_df["id"] = "sample1"  # 添加样本ID
    
    output_f = output_dir / "combined_dataScaleSize.csv"
    data_df.to_csv(output_f, index=False)

def remove_extreme_cells():
    data_dir = output_dir
    data_scale_size_dfs = pd.read_csv(data_dir / "combined_dataScaleSize.csv")
    reg_df = pd.read_csv(data_dir / "otsu_threshold.csv")
    output_f = data_dir / "combined_dataScaleSize_rm_extreme.csv"

    data = data_scale_size_dfs.merge(
        reg_df, left_on=["cellLabel"], right_on=["cell_labels"], how="inner"
    )
    data_rmDAPI = data.query("cell_labels_above").copy()

    log1p_cellSize = np.log1p(data_rmDAPI["cellSize"])
    sigma = 3
    min_cellSize = np.mean(log1p_cellSize) - sigma * np.std(log1p_cellSize)
    max_cellSize = np.mean(log1p_cellSize) + sigma * np.std(log1p_cellSize)
    data_rmDAPI["within_3_sigma"] = data_rmDAPI["cellSize"].apply(
        lambda x: np.expm1(min_cellSize) < x < np.expm1(max_cellSize)
    )
    data_rm_extreme = data_rmDAPI.query("within_3_sigma")

    data_rm_extreme[["cellLabel", "cellSize", "Y_cent", "X_cent"] + MARKER_NAMES].to_csv(output_f, index=False)
    #data_rm_extreme.to_csv(output_f, index=False)

def fix_CD68():
    data_clean_f = "/mnt/nfs/home/wenruiwu/projects/bidmc-jiang-rcc/output/data/20250211_annotation/20250224_combined_dataScaleSize_rm=extreme.csv"
    data_scale_size_dir = "/mnt/nfs/home/wenruiwu/projects/bidmc-jiang-rcc/output/data/20250224_ometiff/data_scale_size"
    output_f = "/mnt/nfs/home/wenruiwu/projects/bidmc-jiang-rcc/output/data/20250211_annotation/20250224_combined_dataScaleSize_rm=extreme_fix=CD68.csv"

    data_clean_df = pd.read_csv(data_clean_f)
    fs = list(Path(data_scale_size_dir).glob("*.csv"))
    data_scale_size_df = pd.concat(
        [pd.read_csv(f).assign(id=f.stem) for f in tqdm(fs, bar_format=TQDM_FORMAT)]
    )

    data_clean_df_fix = data_clean_df.drop(columns=["CD68"]).merge(
        data_scale_size_df[["id", "cellLabel", "cellSize", "DAPI", "CD68"]],
        left_on=["id", "cellLabel", "cellSize"],
        right_on=["id", "cellLabel", "cellSize"],
        how="left",
        suffixes=("", "_y"),
    )
    if np.sum(data_clean_df_fix["DAPI"] - data_clean_df_fix["DAPI_y"] > 1e-6) != 0:
        raise ValueError("DAPI mismatch")
    else:
        data_clean_df_fix = data_clean_df_fix.drop(columns=["DAPI_y"])
    data_clean_df_fix.to_csv(output_f, index=False)

def dapi_normalization():
    data_dir = output_dir
    input_f = data_dir / "combined_dataScaleSize_rm_extreme.csv"
    output_f = data_dir / "combined_dataScaleSize_rm_extreme_normDAPI.csv"

    (data_dir / "figure").mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(input_f)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        data=data, x="1_DAPI", log_scale=(True, False), ax=ax
    )
    ax.set_title("DAPI Distribution Before Normalization")
    fig.savefig(data_dir / "figure" / "02_DAPI_distribution_before_normalization.png")

    median_dapi = data["1_DAPI"].median()
    for marker in tqdm(MARKER_NAMES, desc="Processing", bar_format=TQDM_FORMAT):
        data[marker] /= median_dapi

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        data=data, x="1_DAPI", log_scale=(True, False), ax=ax
    )
    ax.set_title("DAPI Distribution After Normalization")
    fig.savefig(data_dir / "figure" / "02_DAPI_distribution_after_normalization.png")

    data.to_csv(output_f, index=False)

def _arcsinh_transformation(x, cofactor):
    """
    Perform arcsinh transformation on a given value.

    Parameters:
    -----------
    x : float
        Value to transform
    cofactor : float
        Cofactor for transformation

    Returns:
    --------
    float
        Transformed value
    """
    return np.arcsinh(x / cofactor)

def plot_arcsinh_transformations(
    data,
    marker,
    cofactors=[1000, 100, 10, 0.1, 0.01, 0.001, 0.0001],
    hue="id",
    n_row=None,
    legend=False,
    height_sub=6,
    width_sub=8,
):
    """
    Plot multiple arcsinh transformations for given markers

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing marker data
    marker: str
        Marker to transform
    cofactors : list
        List of cofactors for arcsinh transformation
    """
    transformed_data = []
    for cofactor in cofactors:
        transformed = pd.DataFrame(
            {
                "marker": marker,
                "cofactor": cofactor,
                "value": _arcsinh_transformation(data[marker], cofactor),
                "hue": data[hue] if hue is not None else "all",
            }
        )
        transformed_data.append(transformed)
    transformed_df = pd.concat(transformed_data)

    # Create subplot grid
    n_cofactors = len(cofactors)
    if n_row is None:
        n_row = np.sqrt(n_cofactors).astype(int)
    n_col = n_cofactors // n_row + (1 if n_cofactors % n_row != 0 else 0)
    fig, axs = plt.subplots(
        nrows=n_row, ncols=n_col, figsize=(width_sub * n_col, height_sub * n_row)
    )
    axs = axs.flatten()
    for ax in axs[n_cofactors:]:
        ax.axis("off")
    for i, ax in tqdm(
        enumerate(axs[:n_cofactors]),
        desc="Plotting",
        total=n_cofactors,
        bar_format=TQDM_FORMAT,
    ):
        cofactor = cofactors[i]
        marker_data = transformed_df[transformed_df["cofactor"] == cofactor]
        sns.kdeplot(
            data=marker_data,
            x="value",
            hue="hue",
            ax=ax,
            legend=legend,
        )
        ax.set_title(f"{marker} (cofactor={cofactor})")
        ax.set_xlabel("Transformed Value")
    plt.tight_layout()
    plt.close(fig)
    return fig

def arcsinh_transformation():
    data_dir = output_dir
    input_f = data_dir / "combined_dataScaleSize_rm_extreme_normDAPI.csv"
    output_f = data_dir / "combined_dataScaleSize_rm_extreme_normDAPI_arcsinh.csv"

    data = pd.read_csv(input_f)

    markers = MARKER_NAMES
    width_sub = 6
    height_sub = 4
    for marker in markers:
        fig = plot_arcsinh_transformations(
            data,
            marker,
            hue=None,
            n_row=4,
            width_sub=width_sub,
            height_sub=height_sub,
        )
        fig.savefig(data_dir / "figure" / f"03_{marker}_arcsinh_transformations.png")

    cofactor = 0.01
    for marker in tqdm(MARKER_NAMES, desc="Processing", bar_format=TQDM_FORMAT):
        data[marker] = _arcsinh_transformation(data[marker], cofactor)
    data.to_csv(output_f, index=False)

def _quantile_normalization(x, min_quantile, max_quantile, equal_return=0):
    """
    Normalize a given value based on quantiles.

    Parameters:
    -----------
    x : float
        Value to normalize
    min_quantile : float
        Minimum quantile
    max_quantile : float
        Maximum quantile
    equal_return : float
        Value to return when min_val == max_val

    Returns:
    --------
    float
        Normalized value
    """
    min_val = x.quantile(min_quantile)
    max_val = x.quantile(max_quantile)

    # Handle edge case where min and max are equal
    if min_val == max_val:
        return np.full_like(x, equal_return, dtype=float)

    # Normalize values to [0,1] range
    x_norm = (x - min_val) / (max_val - min_val)
    x_norm = np.clip(x_norm, 0, 1)

    return x_norm

def quantile_normalization():
    data_dir = output_dir
    input_f = data_dir / "combined_dataScaleSize_rm_extreme_normDAPI_arcsinh.csv"
    output_f = data_dir / "combined_dataScaleSize_rm_extreme_normDAPI_arcsinh_quantile.csv"
    
    data = pd.read_csv(input_f)
    min_quantile = 0.01
    max_quantile = 0.99

    markers = MARKER_NAMES
    for marker in tqdm(markers, desc="Plotting", bar_format=TQDM_FORMAT):
        x = data[marker]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(x, ax=ax)
        ylim = ax.get_ylim()
        quantiles = [0.005, 0.01, 0.05, 0.95, 0.99, 0.995]
        colors = sns.color_palette(n_colors=len(quantiles))
        for quantile, color in zip(quantiles, colors):
            v_quantile = x.quantile(quantile)
            ax.vlines(
                v_quantile,
                ylim[0],
                ylim[1],
                color=color,
                label=f"{quantile * 100}% ({v_quantile:.2f})",
            )
            ax.legend()
            ax.set_xlabel(marker)
        fig.savefig(data_dir / "figure" / f"04_{marker}_quantile.png")

    for marker in tqdm(MARKER_NAMES, desc="Processing", bar_format=TQDM_FORMAT):
        data[marker] = _quantile_normalization(data[marker], min_quantile, max_quantile)
    data.to_csv(output_f, index=False)


# %%
# id = "RCC-TMA544-dst=reg004-src=reg006"

# seg_mask_geojson_f = (
#     Path(
#         "/mnt/nfs/home/wenruiwu/projects/bidmc-jiang-rcc/output/data/20250116_ometiff/"
#     )
#     / id
#     / "20250125_whole_cell"
#     / "segmentation_mask.geojson"
# )
# with open(seg_mask_geojson_f, "r") as f:
#     seg_mask_geojson = json.load(f)

# data_scale_size_id = data_scale_size_dfs_merge.query(
#     "id == @id").set_index("cellLabel")
# for feature in tqdm(seg_mask_geojson["features"]):
#     name = int(feature["properties"]["name"])
#     classification = feature["properties"]["classification"]
#     if data_scale_size_id.loc[name, "cell_labels_above"]:
#         classification["name"] = "Above OTSU"
#         classification["color"] = [255, 0, 0]
# seg_mask_geojson
# with open(f"{id}_otsu.geojson", "w") as f:
#     json.dump(seg_mask_geojson, f)

# # %%
# seg_mask_f = (
#     Path(
#         "/mnt/nfs/home/wenruiwu/projects/bidmc-jiang-rcc/output/data/20250116_ometiff/"
#     )
#     / id
#     / "20250125_whole_cell"
#     / "segmentation_mask.tiff"
# )
# seg_mask = tifffile.imread(seg_mask_f)
# tifffile.imshow(skimage.segmentation.find_boundaries(seg_mask))
# seg_mask_boundary = skimage.segmentation.find_boundaries(
#     seg_mask, mode="inner")
# tifffile.imwrite("temp_inner.tiff", seg_mask_boundary.astype(bool))
# tifffile.imwrite("temp_inner_uint8.tiff",
#                  (seg_mask_boundary * 255).astype(np.uint8))


# %%
if __name__ == "__main__":
    filter_cell_by_dapi_otsu() #should not be run
    combine_data_scale_size() #should not be run
    remove_extreme_cells()
    # fix_CD68()
    dapi_normalization()
    arcsinh_transformation()
    quantile_normalization()
    None
