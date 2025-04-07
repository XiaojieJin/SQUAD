# utils.py

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import shapiro
from scipy.spatial import ConvexHull
import scrublet as scr
import scanpy as sc
from shapely.geometry import Polygon, box, Point
from collections import defaultdict
from scipy.spatial import distance
from geo_rasterize import rasterize
import random
import time
from scipy.spatial.distance import cosine
from pathlib import Path
import gft_test_5 as gft
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from spatialdata import SpatialData
from sklearn import preprocessing


def polygon_area(coords):
    x = coords["x"]
    y = coords["y"]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def getArea(sdata, polygon_file = None):
    try:
        if 'Area.um2'in sdata.table.obs.columns:
            return sdata.table.obs["Area.um2"]
        elif "volume" in sdata.table.obs.columns:
            groups = polygon_file.groupby('cell_ID')
            area_dict = {}
            for name, group in groups:
                # polygon_coords = [(group.loc[i, "x"], group.loc[i, "y"]) for i in group.index]
                area_dict[name] = polygon_area(group)
            return pd.Series(data=area_dict, index=area_dict.keys(), name="Area")
        elif "cell_area" in sdata.table.obs.columns:
            return sdata.table.obs["cell_area"]
        elif "array_row" in sdata.table.obs.columns:
            cols = [col for col in sdata.shapes.keys() if col.endswith("_008um")]
            col_name = cols[0]
            area_dict = {}
            for name, polygon in sdata.shapes[col_name].iterrows():
                area_dict[name] = polygon.item().area
            return pd.Series(data=area_dict, index=area_dict.keys(), name="Area")
        else:
            return None
    except Exception as e:
        print(f"Error in getArea: {str(e)}")
        raise
def getfiltered_genes_len(sdata):
    try:
        if "negative" in sdata.table.var_names:
            filtered_genes = [gene for gene in sdata.table.var_names if not gene.startswith("negative") and not gene.startswith("systemcontrol")]
            return len(filtered_genes)
        elif "blank" in sdata.table.obsm.keys():
            return (sdata.table.X).shape[1]
        else:
            return (sdata.table.X).shape[1]
    except Exception as e:
        print(f"Error in getfiltered_genes_len: {str(e)}")
        raise

def getfiltered_genes(sdata):
    try:
        if "negative" in sdata.table.var_names:
            filtered_genes = [gene for gene in sdata.table.var_names if not gene.startswith("negative") and not gene.startswith("systemcontrol")]
            # 获取它们在 var_names 中的索引位置
            filtered_gene_indices = [sdata.table.var_names.get_loc(g) for g in filtered_genes]
            # 然后用这些索引去访问 X 的列
            filtered_gene_data = sdata.table.X[:, filtered_gene_indices]
        elif "blank" in sdata.table.obsm.keys():
            filtered_gene_data = sdata.table.X
        elif "MT-" in sdata.table.var_names:
            filtered_genes = [gene for gene in sdata.table.var_names if not gene.startswith("MT-")]
            filtered_gene_indices = [sdata.table.var_names.get_loc(g) for g in filtered_genes]
            filtered_gene_data = sdata.table.X[:, filtered_gene_indices]
        else:
            filtered_gene_data = sdata.table.X
        return filtered_gene_data
    except Exception as e:
        print(f"Error in getfiltered_genes: {str(e)}")
        raise

def getfiltered_neg(sdata):
    try:
        if "negative" in sdata.table.var_names:
            filtered_neg = [gene for gene in sdata.table.var_names if gene.startswith("negative")]
            filtered_neg_indices = [sdata.table.var_names.get_loc(g) for g in filtered_neg]
            filtered_neg_data = sdata.table.X[:, filtered_neg_indices]
        elif "blank" in sdata.table.obsm.keys():
            filtered_neg_data = sdata.table.obsm["blank"]
            filtered_neg_data = sp.csr_matrix(filtered_neg_data)
        elif sdata.table.var_names.str.startswith("MT-").any():
            filtered_neg = [gene for gene in sdata.table.var_names if gene.startswith("MT-")]
            filtered_neg_indices = [sdata.table.var_names.get_loc(g) for g in filtered_neg]
            filtered_neg_data = sdata.table.X[:, filtered_neg_indices]
        else:
            filtered_neg_data = sdata.table.obs["control_probe_counts"] 
        return filtered_neg_data
    except Exception as e:
        print(f"Error in getfiltered_genes: {str(e)}")
        raise
def get_point_list(genes_in_box, cell_name):
    try:
        point_list = []
        if "cellcomp" in genes_in_box.columns:
            #f = open("/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/point_list.csv", "w")
            # Loop through each gene and check if it is inside or outside the red polygon
            for idx, gene in genes_in_box.iterrows():
                gene_point = Point(gene["x_modified_px"], gene["y_modified_px"])
                if gene["cell_ID"] == int(cell_name):
                    point_list.append(gene_point)
                    #f.write(f"{gene['x_modified_px']},{gene['y_modified_px']}\n")
                elif gene["cellcomp"] not in ["Membrane","Nuclear","Cytoplasm"]:
                    point_list.append(gene_point)
                    #f.write(f"{gene['x_modified_px']},{gene['y_modified_px']}\n")
            #f.close()
            return point_list
        else:
            for idx, gene in genes_in_box.iterrows():
                gene_point = Point(gene["x_modified_px"], gene["y_modified_px"])
                point_list.append(gene_point)
            return point_list
    except Exception as e:
        print(f"Error in get_point_list: {str(e)}")
        raise
            
def validate_spatialdata(sdata):
    """验证 SpatialData 对象是否包含必要的字段"""
    required_fields = ['table', 'shapes', 'points']
    for field in required_fields:
        if not hasattr(sdata, field):
            raise ValueError(f"SpatialData object missing required field: {field}")
    return True

# === 图信号处理相关 ===
def gft_r(signal, U):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    return U.T @ signal

def cal_gcc(knee, signal1, signal2, eigenvectors):
    signal1_data = signal1.toarray()
    signal2_data = signal2.toarray()
    gft_signal1 = gft_r(signal1_data, eigenvectors)
    gft_signal2 = gft_r(signal2_data, eigenvectors)
    return 1 - cosine(gft_signal1[1:knee].flatten(), gft_signal2[1:knee].flatten())

def cal_snr(knee_low, knee_high, low_signal, high_signal, eigenvectors_low, eigenvectors_high):
    low_signal_data = low_signal.toarray()
    high_signal_data = high_signal.toarray()
    gft_signal1 = gft_r(low_signal_data, eigenvectors_low)
    gft_signal2 = gft_r(high_signal_data, eigenvectors_high)
    return 10 * np.log10(np.sum(gft_signal1[1:knee_low]**2) / np.sum(gft_signal2[knee_high:-1]**2))


def cal_EOR(sdata, polygon_file=None):
    try:
        validate_spatialdata(sdata)
        if polygon_file is None:
            polygon_file = sdata.shapes
        Area = getArea(sdata, polygon_file)

        if Area.shape[0] != sdata.table.shape[0]:
            # 对 Area 进行行筛选
            matched_cells = Area.index.intersection(sdata.table.obs_names)
            obs_idx = sdata.table.obs_names.get_indexer(matched_cells)
            # 筛选表达矩阵
            filtered_gene_data = getfiltered_genes(sdata)[obs_idx, :]
            filtered_neg_data = getfiltered_neg(sdata)[obs_idx, :]
            Area = Area.loc[matched_cells].to_numpy().reshape((len(matched_cells), 1))
        else:
            filtered_gene_data = getfiltered_genes(sdata)
            filtered_neg_data = getfiltered_neg(sdata)
            Area = Area.to_numpy().reshape((len(sdata.table), 1))
        # Check if the data is a numpy array or a sparse matrix
        # filtered_gene_data = sdata.table[:, filtered_genes].X
        if "control_probe_counts" in sdata.table.obs.columns:
            gene_exp = filtered_gene_data.sum(axis=1)
            neg_exp = filtered_neg_data
            gene_exp = np.asarray(gene_exp)
            neg_exp = np.asarray(neg_exp)
            epsilon = 1e-8
            # print(neg_exp.shape, gene_exp.shape, I.shape, J.shape, Area.shape)
            sdata.table.obs["EOR"] = (
                neg_exp.ravel()
            ) / (
                (gene_exp.ravel() + neg_exp.ravel()) * Area.ravel() + epsilon
            )
        else:
            if isinstance(filtered_gene_data, np.ndarray):
                non_zero_columns_per_cell = np.count_nonzero(filtered_gene_data != 0, axis=1)
            else:
                non_zero_columns_per_cell = np.diff(filtered_gene_data.indptr)

            I = non_zero_columns_per_cell


            if isinstance(filtered_neg_data, np.ndarray):
                non_zero_columns = np.count_nonzero(filtered_neg_data != 0, axis=1)
            else:
                non_zero_columns = np.diff(filtered_neg_data.indptr)
            J = non_zero_columns

            gene_exp = filtered_gene_data.sum(axis=1)
            neg_exp = filtered_neg_data.sum(axis=1)

            I = I.reshape((len(sdata.table), 1))
            J = J.reshape((len(sdata.table), 1))
            gene_exp = np.asarray(gene_exp)
            neg_exp = np.asarray(neg_exp)

            epsilon = 1e-8
            # print(neg_exp.shape, gene_exp.shape, I.shape, J.shape, Area.shape)
            sdata.table.obs["EOR"] = (
                neg_exp.ravel() * I.ravel()
            ) / (
                (gene_exp.ravel() + neg_exp.ravel()) * J.ravel() * Area.ravel() + epsilon
            )
    except Exception as e:
        print(f"Error in cal_EOR: {str(e)}")
        raise

def cal_cell_size(sdata):
    try:
        validate_spatialdata(sdata)
        # data = np.log(getArea(sdata))
        # data = data.tolist()

        # stat, p_value = shapiro(data)

        # alpha = 0.05
        # if p_value > alpha:
        #     mean = np.mean(data)
        #     std_dev = np.std(data)

        #     lower_bound = mean - 3 * std_dev
        #     upper_bound = mean + 3 * std_dev

        #     within_3sigma = (data >= lower_bound) & (data <= upper_bound)
        #     result = np.where(within_3sigma, "pass", "not pass")
        #     sdata.table.obs["cell_size_score"] = result
        # else:
        if "volume" in sdata.table.obs.columns:
            sdata.table.obs["cell_size_score"] = sdata.table.obs["volume"]
        else:
            sdata.table.obs["cell_size_score"] = getArea(sdata)
    except Exception as e:
        print(f"Error in cal_cell_size: {str(e)}")
        raise

def cal_sensitivity_saturation(sdata, polygon_file = None):
    try:
        validate_spatialdata(sdata)
        # filtered_genes = [gene for gene in sdata.table.var_names if not gene.startswith("negative") and not gene.startswith("systemcontrol")]
        # # 获取它们在 var_names 中的索引位置
        # filtered_gene_indices = [sdata.table.var_names.get_loc(g) for g in filtered_genes]
        # # 然后用这些索引去访问 X 的列
        # filtered_gene_data = sdata.table.X[:, filtered_gene_indices]
        filtered_gene_data = getfiltered_genes(sdata)
        if isinstance(filtered_gene_data, np.ndarray):
            non_zero_columns_per_cell = np.count_nonzero(filtered_gene_data != 0, axis=1)
        else:
            non_zero_columns_per_cell = np.diff(filtered_gene_data.indptr)

        trans = non_zero_columns_per_cell
        trans = trans.reshape((len(sdata.table), 1))
        umi = filtered_gene_data.sum(axis=1)
        umi = np.asarray(umi)
        umi = umi.reshape((len(sdata.table), 1))
        Area = getArea(sdata, polygon_file)
        Area = Area.to_numpy().reshape((len(sdata.table), 1))
        trans_total = getfiltered_genes_len(sdata)

        sdata.table.obs["sensitivity_1"] = trans/Area
        sdata.table.obs["sensitivity_2"] = trans
        sdata.table.obs["sensitivity_3"] = trans/(trans_total*Area)

        sdata.table.obs["saturation_1"] = umi/Area
        sdata.table.obs["saturation_2"] = umi
    except Exception as e:
        print(f"Error in cal_sensitivity_saturation: {str(e)}")
        raise

def cal_solidity_circularity(sdata, polygon_file = None):
    try:
        validate_spatialdata(sdata)
        
        # # 创建索引映射字典
        # id_mapping = dict(zip(sdata.table.obs_names, sdata_cell_ids))
        if polygon_file is None:
            if "cell_boundaries" in sdata.shapes.keys():
                sdata_cell_ids = sdata.table.obs["cell_id"]
                # 确保索引格式一致
                sdata_cell_ids = [str(id) for id in sdata_cell_ids]
                polygon_file = sdata.shapes["cell_boundaries"]
            else:
                sdata_cell_ids = sdata.table.var_names
                # 确保索引格式一致
                sdata_cell_ids = [str(id) for id in sdata_cell_ids]
            
                cols = [col for col in sdata.shapes.keys() if col.endswith("_008um")]
                col_name = cols[0]
                polygon_file = sdata.shapes[col_name]
            output_dict = {}
            output_dict["name"] = []
            output_dict["solidity"] = []
            output_dict["circularity"] = []
            for index, row in polygon_file.iterrows():
                output_dict["name"].append(index)
                polygon = row.item()

                area = polygon.area
                perimeter = polygon.length
                area = polygon.area
                perimeter = polygon.length

                polygon_coords = []
                for x, y in polygon.exterior.coords:
                    polygon_coords.append((x, y))
                hull = ConvexHull(polygon_coords)
                hull_area = hull.volume
                
                solidity = area / hull_area if hull_area > 0 else 0
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                output_dict["solidity"].append(solidity)
                output_dict["circularity"].append(circularity)
                
        else:
            # 从sdata.table的索引中提取_前的数字
            sdata_cell_ids = [idx.split('_')[0] for idx in sdata.table.obs_names]
            
            # 确保索引格式一致
            sdata_cell_ids = [str(id) for id in sdata_cell_ids]
            polygon_file['cell_ID'] = polygon_file['cell_ID'].astype(str)     
            groups = polygon_file.groupby('cell_ID')
            output_dict = {}
            output_dict["name"] = []
            output_dict["solidity"] = []
            output_dict["circularity"] = []
            # # 添加调试信息
            # print("Debug: Number of cells in polygon_file:", len(polygon_file['cell_ID'].unique()))
            # print("Debug: Number of cells in sdata.table:", len(sdata.table.obs_names))
            # print("Debug: First few cell IDs in polygon_file:", polygon_file['cell_ID'].unique()[:5])
            # print("Debug: First few cell IDs in sdata.table (after processing):", sdata_cell_ids[:5])
            
            for name, group in groups:
                output_dict["name"].append(name)
                if "x_local_px" in group.columns:
                    polygon_coords = [(group.loc[i, "x_local_px"], group.loc[i, "y_local_px"]) for i in group.index]
                else:
                    polygon_coords = [(group.loc[i, "x"], group.loc[i, "y"]) for i in group.index]
                if len(polygon_coords) < 3:
                    print(f"Debug: Cell {name} has less than 3 coordinates")
                    return None, None
            
                polygon = Polygon(polygon_coords)
                area = polygon.area
                perimeter = polygon.length
                hull = ConvexHull(polygon_coords)
                hull_area = hull.volume
                
                solidity = area / hull_area if hull_area > 0 else 0
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                output_dict["solidity"].append(solidity)
                output_dict["circularity"].append(circularity)
                
        out_df = pd.DataFrame(output_dict)
        out_df = out_df.set_index("name")
        
        # # 添加调试信息
        # print("Debug: Number of cells in out_df before reindex:", len(out_df))
        # print("Debug: First few indices in out_df:", out_df.index[:5])
        
        # # 创建反向映射字典
        # reverse_mapping = {v: k for k, v in id_mapping.items()}
        
        # # 使用映射进行重索引
        # out_df = out_df.reindex([reverse_mapping.get(idx, idx) for idx in out_df.index])
        out_df = out_df.reindex(sdata_cell_ids)
        
        # # 添加调试信息
        # print("Debug: Number of cells in out_df after reindex:", len(out_df))
        # print("Debug: Number of NA values in solidity:", out_df["solidity"].isna().sum())
        # print("Debug: First few cell IDs in out_df:", out_df.index[:5])
        # print("Debug: First few cell IDs in sdata.table (after processing):", sdata.table.obs_names[:5])
        out_df.reset_index(drop=True, inplace=True)
        sdata.table.obs["solidity"] = out_df["solidity"].values
        sdata.table.obs["circularity"] = out_df["circularity"].values
    except Exception as e:
        print(f"Error in cal_solidity_circularity: {str(e)}")
        raise

def get_score_list(gene_df, polygon_df, cell_name, bin_size = 25, ratio_neighbors = 1, eigenvectors = None, knee = 0, box_extend_ratio = 0):
    tmp_ratio_neighbors = ratio_neighbors
    result_list = []
    # Record the start time
    start_time = time.time()

    # Group the data by 'fov_cell' (each group represents a polygon)
    groups = polygon_df.groupby('cell_ID')

    # Set the specific idx (fov_cell) that you want to select
    specific_fov_cell = cell_name

    # Filter the DataFrame to get the specific polygon by 'fov_cell'
    specific_polygon_group = polygon_df[polygon_df['cell_ID'] == specific_fov_cell]
    # Extract x and y coordinates for the current polygon
    x_coords = specific_polygon_group['x']
    y_coords = specific_polygon_group['y']

    # Append the first point to close the polygon
    x_coords = list(x_coords) + [x_coords.iloc[0]]
    y_coords = list(y_coords) + [y_coords.iloc[0]]

    # Create a shapely Polygon object from the coordinates
    polygon = Polygon(zip(x_coords, y_coords))
    
    minx_ori, miny_ori, maxx_ori, maxy_ori = polygon.bounds # Original box coordinates
    x_range = maxx_ori - minx_ori
    y_range = maxy_ori - miny_ori
    
    if box_extend_ratio == 0:
        minx0 = minx_ori
        maxx0 = maxx_ori
        miny0 = miny_ori
        maxy0 = maxy_ori
    else:
        minx0 = minx_ori - x_range * box_extend_ratio # Extend box coordinates
        maxx0 = maxx_ori + x_range * box_extend_ratio
        miny0 = miny_ori - y_range * box_extend_ratio
        maxy0 = maxy_ori + y_range * box_extend_ratio        

    x_coords_2 = []
    y_coords_2 = []
    for x in x_coords:
        x_coords_2.append((x - minx0)/((maxx0-minx0)/bin_size))
    for y in y_coords:
        y_coords_2.append((y - miny0)/((maxy0-miny0)/bin_size))

    # Modify the coordinates (example: adding 1 to each x-coordinate)
    modified_coords = [((x - minx0)/((maxx0-minx0)/bin_size), (y - miny0)/((maxy0-miny0)/bin_size)) for x, y in polygon.exterior.coords]

    # Create a new polygon with modified coordinates
    modified_polygon = Polygon(modified_coords)

    # Convert the shapely Polygon to a list of shapes (as geo-rasterize expects)
    shapes = [modified_polygon]
    
    # Foreground value for the red polygon (e.g., 1 for cells marked in red)
    foregrounds = [1]

    # Set the size of the raster grid (50x50)
    raster_size = (bin_size, bin_size)

    # Rasterize the polygon
    rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='replace')

    flattened_array = rasterized.flatten()

    # Step 2: Convert to a pandas DataFrame with a single column named 'mask'
    df = pd.DataFrame(flattened_array, columns=["mask"])

    filtered_sparse = sp.csr_matrix(df)

    tmp_adata = ad.AnnData(filtered_sparse)

    if box_extend_ratio == 0:
        # Get the bounding box of the polygon
        minx, miny, maxx, maxy = modified_polygon.bounds
    else:
        minx = minx0
        maxx = maxx0
        miny = miny0
        maxy = maxy0

    # Calculate bin width and height
    bin_width = (maxx - minx) / raster_size[1]
    bin_height = (maxy - miny) / raster_size[0]
    
    # Create a list to store centroids
    centroids = {"centroid_x":[],"centroid_y":[]}

    # Draw grid (bins) lines and color the touched bins in light blue
    for i in range(raster_size[0]):
        for j in range(raster_size[1]):

            # Coordinates of the bin
            bin_x_min = minx + j * bin_width
            bin_x_max = bin_x_min + bin_width
            bin_y_min = miny + i * bin_height
            bin_y_max = bin_y_min + bin_height

            # Calculate the centroid of the bin
            centroid_x = (bin_x_min + bin_x_max) / 2
            centroid_y = (bin_y_min + bin_y_max) / 2

            centroids["centroid_x"].append(centroid_x)
            centroids["centroid_y"].append(centroid_y)
    centroids_df = pd.DataFrame(centroids)
    x_global_px_array = centroids_df['centroid_x'].to_numpy().reshape(-1, 1)
    y_global_px_array = centroids_df['centroid_y'].to_numpy().reshape(-1, 1)

    tmp_adata.obsm['array_row']  = x_global_px_array
    tmp_adata.obsm["array_col"] = y_global_px_array
    
    if eigenvectors is None:
        low_freq, high_freq = gft.determine_frequency_ratio(tmp_adata, ratio_neighbors = tmp_ratio_neighbors)
        eigenvectors = tmp_adata.uns['FMs_after_select']["low_FMs"]
        eigenvalues = tmp_adata.uns['FMs_after_select']['low_FMs_frequency']
        
        knee = low_freq

    ########################################

    # Filter and draw the genes that fall within the bounding box of the red polygon
    
    genes_in_box = gene_df[
        (gene_df['x'] >= minx0) &
        (gene_df['x'] <= maxx0) &
        (gene_df['y'] >= miny0) &
        (gene_df['y'] <= maxy0)
    ]

    #f = open("/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/point_list.csv", "w")
    # Classify genes as inside or outside the polygon
    genes_in_box["x_modified_px"] = (genes_in_box['x'] - minx0)/((maxx0-minx0)/bin_size)
    genes_in_box["y_modified_px"] = (genes_in_box['y'] - miny0)/((maxy0-miny0)/bin_size)

    point_list = get_point_list(genes_in_box, cell_name)
    shapes = point_list
    
    # Foreground value for the red polygon (e.g., 1 for cells marked in red)
    foregrounds = [1 for i in range(len(point_list))]

    # Set the size of the raster grid (50x50)
    raster_size = (bin_size, bin_size)

    # Rasterize the polygon
    rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='add')
    flattened_array = rasterized.flatten()

    # Step 2: Convert to a pandas DataFrame with a single column named 'mask'
    df = pd.DataFrame(flattened_array, columns=["mask"])
    filtered_sparse = sp.csr_matrix(df)
    tmp_adata2 = ad.AnnData(filtered_sparse)
    tmp_adata2.obsm['array_row']  = x_global_px_array
    tmp_adata2.obsm["array_col"] = y_global_px_array
    
    cos_similarity = cal_gcc(knee, tmp_adata.X, tmp_adata2.X, eigenvectors)
    return cos_similarity, eigenvectors, knee

def get_score_list_trans(gene_df, gene_name, x_coords, y_coords, bin_size = 25, ratio_neighbors = 1, eigenvectors_low = None, knee_low = 0, eigenvectors_high = None, knee_high = 0):
    start_time_get_adata = time.time()
    
    tmp_ratio_neighbors = ratio_neighbors
    # Append the first point to close the polygon
    x_coords = list(x_coords) + [x_coords[0]]
    y_coords = list(y_coords) + [y_coords[0]]

    # Create a shapely Polygon object from the coordinates
    polygon = Polygon(zip(x_coords, y_coords))
    minx0, miny0, maxx0, maxy0 = polygon.bounds
    
    x_coords_2 = []
    y_coords_2 = []
    for x in x_coords:
        x_coords_2.append((x - minx0)/((maxx0-minx0)/bin_size))
    for y in y_coords:
        y_coords_2.append((y - miny0)/((maxy0-miny0)/bin_size))

    # Modify the coordinates (example: adding 1 to each x-coordinate)
    modified_coords = [((x - minx0)/((maxx0-minx0)/bin_size), (y - miny0)/((maxy0-miny0)/bin_size)) for x, y in polygon.exterior.coords]

    # Create a new polygon with modified coordinates
    modified_polygon = Polygon(modified_coords)

    # Convert the shapely Polygon to a list of shapes (as geo-rasterize expects)
    shapes = [modified_polygon]
    
    # Foreground value for the red polygon (e.g., 1 for cells marked in red)
    foregrounds = [1]

    # Set the size of the raster grid (50x50)
    raster_size = (bin_size, bin_size)

    # Rasterize the polygon
    rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='replace')

    flattened_array = rasterized.flatten()

    # Step 2: Convert to a pandas DataFrame with a single column named 'mask'
    df = pd.DataFrame(flattened_array, columns=["mask"])

    filtered_sparse = sp.csr_matrix(df)

    tmp_adata = ad.AnnData(filtered_sparse)

    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = modified_polygon.bounds

    # Calculate bin width and height
    bin_width = (maxx - minx) / raster_size[1]
    bin_height = (maxy - miny) / raster_size[0]
    
    # Create a list to store centroids
    centroids = {"centroid_x":[],"centroid_y":[]}

    # Draw grid (bins) lines and color the touched bins in light blue
    for i in range(raster_size[0]):
        for j in range(raster_size[1]):

            # Coordinates of the bin
            bin_x_min = minx + j * bin_width
            bin_x_max = bin_x_min + bin_width
            bin_y_min = miny + i * bin_height
            bin_y_max = bin_y_min + bin_height

            # Calculate the centroid of the bin
            centroid_x = (bin_x_min + bin_x_max) / 2
            centroid_y = (bin_y_min + bin_y_max) / 2

            centroids["centroid_x"].append(centroid_x)
            centroids["centroid_y"].append(centroid_y)
    centroids_df = pd.DataFrame(centroids)
    x_global_px_array = centroids_df['centroid_x'].to_numpy().reshape(-1, 1)
    y_global_px_array = centroids_df['centroid_y'].to_numpy().reshape(-1, 1)

    tmp_adata.obsm['array_row']  = x_global_px_array
    tmp_adata.obsm["array_col"] = y_global_px_array
    
    if (eigenvectors_low is None) and (eigenvectors_high is None):
        start_time_GFT = time.time()
    
        low_freq, high_freq = gft.determine_frequency_ratio(tmp_adata, ratio_neighbors = tmp_ratio_neighbors)
        eigenvectors_low = tmp_adata.uns['FMs_after_select']["low_FMs"]
        eigenvalues_low = tmp_adata.uns['FMs_after_select']['low_FMs_frequency']
        
        eigenvectors_high = tmp_adata.uns['FMs_after_select']["high_FMs"]
        eigenvalues_high = tmp_adata.uns['FMs_after_select']['high_FMs_frequency']
        
        knee_low = low_freq
        knee_high = high_freq

    ########################################
    
    start_time_get_adata_2 = time.time()
    
    start_time_point_list = time.time()
    point_list = []
    genes_in_box = gene_df
    genes_in_box["x_modified_px"] = (genes_in_box['x'] - minx0)/((maxx0-minx0)/bin_size)
    genes_in_box["y_modified_px"] = (genes_in_box['y'] - miny0)/((maxy0-miny0)/bin_size)
    if "target" in gene_df.columns:
        filtered_genes = genes_in_box[genes_in_box["target"] == gene_name]
    else:
        filtered_genes = genes_in_box[genes_in_box["gene"] == gene_name]
    point_array = np.column_stack((filtered_genes["x_modified_px"], filtered_genes["y_modified_px"]))
    point_list = [Point(x, y) for x, y in point_array]
    
    start_time_get_X = time.time()
                              
    # Convert the shapely Polygon to a list of shapes (as geo-rasterize expects)
    shapes = point_list
    
    # Foreground value for the red polygon (e.g., 1 for cells marked in red)
    foregrounds = [1 for i in range(len(point_list))]

    # Set the size of the raster grid (50x50)
    raster_size = (bin_size, bin_size)

    # Rasterize the polygon
    rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='add')
    flattened_array = rasterized.flatten()

    # Step 2: Convert to a pandas DataFrame with a single column named 'mask'
    df = pd.DataFrame(flattened_array, columns=["mask"])
    filtered_sparse = sp.csr_matrix(df)
    tmp_adata2 = ad.AnnData(filtered_sparse)
    tmp_adata2.obsm['array_row']  = x_global_px_array
    tmp_adata2.obsm["array_col"] = y_global_px_array
    
    start_time_cal_snr = time.time()
    snr = cal_snr(knee_low, knee_high, tmp_adata2.X, tmp_adata2.X, eigenvectors_low, eigenvectors_high)
    
    return snr, eigenvectors_low, eigenvectors_high, knee_low, knee_high

def cal_sgcc(sdata, bin_size = 60, ratio_neighbors = 10, box_extend_ratio = 0, polygon_file = None):
    """
    计算空间基因共表达系数
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing expression data
    output_prefix : str, optional
        Prefix for output files
    """
    try:
        validate_spatialdata(sdata)

        tmp_bin_size = bin_size
        tmp_ratio_neighbors = ratio_neighbors


        
        if polygon_file is None:
            sdata_cell_ids = sdata.table.obs["cell_id"]
            # 确保索引格式一致
            sdata_cell_ids = [str(id) for id in sdata_cell_ids]
            if "cell_boundaries" in sdata.shapes.keys():
                polygon_file = sdata.shapes["cell_boundaries"]
            else:
                cols = [col for col in sdata.shapes.keys() if col.endswith("_008um")]
                col_name = cols[0]
                polygon_file = sdata.shapes[col_name]

            polygon_df = polygon_file
            points_df = sdata.points["transcripts"]
            # 转换成 pandas DataFrame（如果你想要）
            #gene_df = points_df.compute()

            gene_df = points_df
            rows = []
            for cell_id, polygon in polygon_df.iterrows():
                for x, y in polygon.item().exterior.coords:  # 取 Polygon 的外部边界点
                    rows.append({'cell_ID': cell_id, 'x': x, 'y': y})

            polygon_df = pd.DataFrame(rows)
            # output_dict = {}
                
            # output_dict["name"] = []
            # output_dict["score"] = []
            # flag = 0
            # for index, row in polygon_file.iterrows():
            #     output_dict["name"].append(index)

            #     #print(f"We are calculating sgcc, bs = {tmp_bin_size}, k = {tmp_ratio_neighbors}")
            #     print(f"For cell: {index}: ")
            #     if flag == 0:
            #         result_g, tmp_eigenvectors, tmp_knee = get_score_list(gene_df, polygon_df, index, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors, box_extend_ratio = 0.1)
            #     else:
            #         result_g, tmp_eigenvector2, tmp_knee_2 = get_score_list(gene_df, polygon_df, index, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors, eigenvectors = tmp_eigenvectors, knee = tmp_knee, box_extend_ratio = 0.1)
            #     output_dict["score"].append(result_g)
            #     flag += 1
        else:
            sdata_cell_ids = [idx.split('_')[0] for idx in sdata.table.obs_names]
            # 确保索引格式一致
            sdata_cell_ids = [str(id) for id in sdata_cell_ids]
            polygon_file['cell_ID'] = polygon_file['cell_ID'].astype(str)
            
            # 获取唯一 key 名
            points_key = list(sdata.points.keys())[0]

            # 获取对应的 Dask DataFrame
            points_df = sdata.points[points_key]

            # 转换成 pandas DataFrame（如果你想要）
            gene_df = points_df.compute()
            if "cellcomp" in gene_df.columns:
                gene_df["cellcomp"] = gene_df["cellcomp"].fillna("Unknown")
            

            polygon_df = polygon_file
            if "x_local_px" in polygon_df.columns:
                polygon_df.rename(columns={"x_local_px": "x"}, inplace=True)
                polygon_df.rename(columns={"y_local_px": "y"}, inplace=True)
            
            
        groups = polygon_df.groupby('cell_ID')
        output_dict = {}
            
        output_dict["name"] = []
        output_dict["score"] = []
        flag = 0
        for name, group in groups:
            output_dict["name"].append(name)

            #print(f"We are calculating sgcc, bs = {tmp_bin_size}, k = {tmp_ratio_neighbors}")
            print(f"For cell: {name}: ")
            if flag == 0:
                result_g, tmp_eigenvectors, tmp_knee = get_score_list(gene_df, polygon_df, name, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors, box_extend_ratio = 0.1)
            else:
                result_g, tmp_eigenvector2, tmp_knee_2 = get_score_list(gene_df, polygon_df, name, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors, eigenvectors = tmp_eigenvectors, knee = tmp_knee, box_extend_ratio = 0.1)
            output_dict["score"].append(result_g)
            flag += 1
        out_df = pd.DataFrame(output_dict)
        out_df = out_df.set_index("name")
        # Reindex the metadata to align with adata.obs_names
        # This ensures that the metadata aligns with the observations in the AnnData object
        out_df = out_df.reindex(sdata_cell_ids)
        sdata.table.obs["sgcc"] = out_df["score"].values
            
    except Exception as e:
        print(f"Error in cal_sgcc: {str(e)}")
        raise

def cal_signal_noise_ratio(sdata, bin_size = 60, ratio_neighbors = 10, polygon_file = None):
    """
    计算信噪比
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing expression data
    output_prefix : str, optional
        Prefix for output files
    """
    try:
        validate_spatialdata(sdata)
    
        tmp_bin_size = bin_size
        tmp_ratio_neighbors = ratio_neighbors
        if polygon_file is None:
            if "cell_boundaries" in sdata.shapes.keys():
                polygon_file = sdata.shapes["cell_boundaries"]
            else:
                cols = [col for col in sdata.shapes.keys() if col.endswith("_008um")]
                col_name = cols[0]
                polygon_file = sdata.shapes[col_name]

            polygon_df = polygon_file
            points_df = sdata.points["transcripts"]
            # 转换成 pandas DataFrame（如果你想要）
            gene_df = points_df.compute()
            rows = []
            for cell_id, polygon in polygon_df.iterrows():
                for x, y in polygon.item().exterior.coords:  # 取 Polygon 的外部边界点
                    rows.append({'cell_ID': cell_id, 'x': x, 'y': y})

            polygon_df = pd.DataFrame(rows)
            groups =  gene_df.groupby('feature_name')
            # output_dict = {}
            # output_dict["name"] = []
            # output_dict["score"] = []

            # x_coords = np.array([gene_df['x'].min(),gene_df['x'].min(),gene_df['x'].max(),gene_df['x'].max()])
            # y_coords = np.array([gene_df['y'].min(),gene_df['y'].max(),gene_df['y'].max(),gene_df['x'].min()])
            # flag = 0
            # for index, row in polygon_file.iterrows():
            #     output_dict["name"].append(index)
            #     if flag == 0:
            #         result_g, tmp_eigenvectors_low, tmp_eigenvectors_high, tmp_knee_low, tmp_knee_high = get_score_list_trans(gene_df, index, x_coords, y_coords, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors)
            #     else:
            #         result_g, tmp_eigenvector_low_2, tmp_eigenvectors_high_2, tmp_knee_low_2, tmp_knee_high_2 = get_score_list_trans(gene_df, index, x_coords, y_coords, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors, eigenvectors_low = tmp_eigenvectors_low, knee_low = tmp_knee_low, eigenvectors_high = tmp_eigenvectors_high, knee_high = tmp_knee_high)
            #     output_dict["score"].append(result_g)
            #     flag += 1
        else:
            # 获取唯一 key 名
            points_key = list(sdata.points.keys())[0]

            # 获取对应的 Dask DataFrame
            points_df = sdata.points[points_key]

            # 转换成 pandas DataFrame（如果你想要）
            gene_df = points_df.compute()
            if "cellcomp" in gene_df.columns:
                gene_df["cellcomp"] = gene_df["cellcomp"].fillna("Unknown")

            polygon_df = polygon_file
            polygon_df.rename(columns={"x_local_px": "x"}, inplace=True)
            polygon_df.rename(columns={"y_local_px": "y"}, inplace=True)

            if "target" in gene_df.columns:
                groups =  gene_df.groupby('target')
                x_coords = np.array([gene_df['x'].min(),gene_df['x'].min(),gene_df['x'].max(),gene_df['x'].max()])
                y_coords = np.array([gene_df['y'].min(),gene_df['y'].max(),gene_df['y'].max(),gene_df['x'].min()])
            else:
                groups =  gene_df.groupby('gene')
                x_coords = np.array([gene_df['x'].min(),gene_df['x'].min(),gene_df['x'].max(),gene_df['x'].max()])
                y_coords = np.array([gene_df['y'].min(),gene_df['y'].max(),gene_df['y'].max(),gene_df['x'].min()])
            
        output_dict = {}
        output_dict["name"] = []
        output_dict["score"] = []
        flag = 0
        
        for name, group in groups:
            output_dict["name"].append(name)
            if flag == 0:
                result_g, tmp_eigenvectors_low, tmp_eigenvectors_high, tmp_knee_low, tmp_knee_high = get_score_list_trans(gene_df, name, x_coords, y_coords, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors)
            else:
                result_g, tmp_eigenvector_low_2, tmp_eigenvectors_high_2, tmp_knee_low_2, tmp_knee_high_2 = get_score_list_trans(gene_df, name, x_coords, y_coords, bin_size = tmp_bin_size, ratio_neighbors = tmp_ratio_neighbors, eigenvectors_low = tmp_eigenvectors_low, knee_low = tmp_knee_low, eigenvectors_high = tmp_eigenvectors_high, knee_high = tmp_knee_high)
            output_dict["score"].append(result_g)
            flag += 1

        out_df = pd.DataFrame(output_dict)
        out_df = out_df.set_index("name")
        out_df = out_df.reindex(sdata.table.var_names)
        sdata.table.var["snr"] = out_df["score"]
            
    except Exception as e:
        print(f"Error in cal_signal_noise_ratio: {str(e)}")
        raise

def cal_scrublet(sdata):
    """计算双胞体得分
    
    Raises:
        ValueError: 当SpatialData对象无效时
        Exception: 当计算过程中出错时
    """
    try:
        print("开始计算双胞体得分")
        tmp_data = getfiltered_genes(sdata)
        # 确保使用稀疏矩阵，节省内存，避免线程爆炸
        if isinstance(tmp_data, np.ndarray):
            sparse_matrix = sp.csr_matrix(tmp_data)
            # sparse_matrix = sdata.table.X.tocsr()
        else:
            sparse_matrix = tmp_data.tocsr()

        # Scrublet 要求的是 cells x genes 的 count matrix，使用 csr/csc 都行
        counts_matrix = sparse_matrix

        # n_components = min(30, counts_matrix.shape[0], counts_matrix.shape[1])
        # 创建 Scrublet 实例
        scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=0.3, sim_doublet_ratio=30)
        doublet_scores, predicted_doublets = scrub.scrub_doublets()
        sdata.table.obs["scrublet_1"] = doublet_scores
        sdata.table.obs['predicted_doublets'] = predicted_doublets
        print("双胞体得分计算完成")
    except Exception as e:
        print(f"计算双胞体得分时出错: {str(e)}")
        raise

def calculate_quality_score(sdata, feature_columns, output_prefix=None, prob_threshold=0.05):
    """
    计算质量得分
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing expression data
    feature_columns : list
        List of feature columns to use for quality score calculation
    output_prefix : str, optional
        Prefix for output files
    prob_threshold : float, optional
        Probability threshold for quality score calculation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with quality scores
    """
    try:
        validate_spatialdata(sdata)
        
        # 保存原始细胞顺序
        if "cell_ID" in sdata.table.obs.columns:
            original_cell_order = sdata.table.obs['cell_ID'].values
        elif "cell_id" in sdata.table.obs.columns:
            original_cell_order = sdata.table.obs["cell_id"]
            # 确保索引格式一致
            original_cell_order = [str(id) for id in original_cell_order]
        else:
            original_cell_order = sdata.table.obs_names

        # 验证细胞名称是否唯一
        if len(original_cell_order) != len(set(original_cell_order)):
            raise ValueError("Warning: Duplicate cell names found in the data!")

        # 验证所有特征列是否存在于数据框中
        missing_columns = [col for col in feature_columns if col not in sdata.table.obs.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")

        # 提取特征数据
        X = sdata.table.obs[feature_columns].values

        # 创建并训练GMM模型
        n_components = 2  # 使用2个高斯分布
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)

        # 计算每个样本属于每个高斯分布的概率
        probs = gmm.predict_proba(X)

        # 获取每个样本属于主要高斯分布的概率
        main_component_probs = probs[:, np.argmax(gmm.weights_)]

        # 将结果添加到原始数据框
        sdata.table.obs['probability'] = main_component_probs

        # # 创建概率分布图
        # plt.figure(figsize=(10, 6))
        # plt.hist(main_component_probs, bins=50, alpha=0.7, density=True)
        # plt.title('Distribution of Conditional Probabilities')
        # plt.xlabel('Probability')
        # plt.ylabel('Density')

        # # 添加均值和中位数文本
        # mean_val = np.mean(main_component_probs)
        # median_val = np.median(main_component_probs)
        # plt.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}', 
        #         transform=plt.gca().transAxes, verticalalignment='top')

        # plt.tight_layout()
        # plt.savefig('probability_distributions4_multivariate.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # # 创建边缘分布图
        # n_features = len(feature_columns)
        # n_rows = (n_features + 2) // 3
        # plt.figure(figsize=(15, 4*n_rows))

        # # 获取主要高斯分布的索引
        # main_component_idx = np.argmax(gmm.weights_)

        # for idx, feature in enumerate(feature_columns):
        #     plt.subplot(n_rows, 3, idx + 1)
            
        #     # 获取当前特征的边缘分布参数
        #     feature_idx = feature_columns.index(feature)
        #     means = gmm.means_[:, feature_idx]
        #     covars = gmm.covariances_[:, feature_idx, feature_idx]
        #     weights = gmm.weights_
            
        #     # 生成x轴点
        #     x = np.linspace(np.min(X[:, feature_idx]), np.max(X[:, feature_idx]), 200)
            
        #     # 绘制原始数据直方图
        #     plt.hist(X[:, feature_idx], bins=50, density=True, alpha=0.7, label='Raw Data')
            
        #     # 绘制每个高斯分量
        #     for i in range(n_components):
        #         if i == main_component_idx:
        #             color = 'orange'  # 主要高斯分布用橘色
        #             label = 'Main Gaussian'
        #         else:
        #             color = 'green'   # 次要高斯分布用绿色
        #             label = 'Secondary Gaussian'
        #         pdf = weights[i] * (1/np.sqrt(2*np.pi*covars[i])) * np.exp(-(x-means[i])**2/(2*covars[i]))
        #         plt.plot(x, pdf, '--', color=color, label=label)
            
        #     plt.title(f'Marginal Distribution of {feature}')
        #     plt.xlabel('Value')
        #     plt.ylabel('Density')
        #     plt.legend()

        # plt.tight_layout()
        # plt.savefig('raw_distributions_with_gmm4_multivariate.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # 添加quality列
        sdata.table.obs['quality'] = (sdata.table.obs['probability'] > 0.05).astype(int)

        # # 选择要保存的列
        # output_columns = ['fov_cell'] + feature_columns + ['probability', 'quality']

        # # 验证最终细胞顺序与原始顺序匹配
        # if not np.array_equal(df['fov_cell'].values, original_cell_order):
        #     raise ValueError("Cell order has changed during processing!")

        # # 保存结果到CSV文件
        # df[output_columns].to_csv('obs_result_fov122_with_probs4_multivariate.csv', index=False)
            
        return 0
        
    except Exception as e:
        print(f"Error in calculate_quality_score: {str(e)}")
        raise
