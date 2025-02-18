import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.stats import shapiro
import scrublet as scr
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import os
import argparse
from scipy.spatial import ConvexHull
import gft_test_4 as gft
import gft_test_5 as gft_5
from shapely.geometry import Polygon, box, Point
from collections import defaultdict
from scipy.spatial import distance
from geo_rasterize import rasterize
import random
import time
from scipy.spatial.distance import cosine
from pathlib import Path

def cal_EOR(adata):
    Area = adata.obs["Area.um2"]
    filtered_genes = [gene for gene in adata.var_names if not gene.startswith("Negative") and not gene.startswith("SystemControl")]
    filtered_gene_data = adata[:, filtered_genes].X
    # Check if the data is a numpy array or a sparse matrix
    if isinstance(filtered_gene_data, np.ndarray):
        # For numpy array, count non-zero entries in each row (per cell)
        non_zero_columns_per_cell = np.count_nonzero(filtered_gene_data != 0, axis=1)
    else:
        # For sparse matrix, count non-zero entries in each row (per cell)
        non_zero_columns_per_cell = np.diff(filtered_gene_data.indptr)

    # Now, `non_zero_columns_per_cell` is a vector with the number of non-zero columns per cell
    I = non_zero_columns_per_cell

    filtered_neg = [neg for neg in adata.var_names if neg.startswith("Negative")]
    filtered_neg_data = adata[:, filtered_neg].X
    if isinstance(filtered_neg_data, np.ndarray):
        non_zero_columns = np.count_nonzero(filtered_neg_data != 0, axis=1)
    else:
        non_zero_columns = np.diff(filtered_neg_data.indptr)
    J = non_zero_columns

    gene_exp = filtered_gene_data.sum(axis=1)
    neg_exp = filtered_neg_data.sum(axis=1)
    #gene_exp = gene_exp.reshape(-1)
    #neg_exp = neg_exp.reshape(-1)

    I = I.reshape((adata.n_obs, 1))
    J = J.reshape((adata.n_obs, 1))
    gene_exp = np.asarray(gene_exp)
    neg_exp = np.asarray(neg_exp)
    Area = Area.to_numpy().reshape((adata.n_obs, 1))
    epsilon = 1e-8
    adata.obs["EOR"] = (neg_exp * I)/((gene_exp+neg_exp) * J * Area + epsilon)

def cal_cell_size(adata):
    # Your data list
    data = np.log(adata.obs["Area.um2"])
    data = data.tolist()

    # Perform the Shapiro-Wilk test
    stat, p_value = shapiro(data)

    print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p_value}")

    # Interpretation
    alpha = 0.05
    if p_value > alpha:
        mean = np.mean(data)
        std_dev = np.std(data)

        # Define the lower and upper bounds for 3-sigma
        lower_bound = mean - 3 * std_dev
        upper_bound = mean + 3 * std_dev

        # Check which elements are within the 3-sigma range
        within_3sigma = (data >= lower_bound) & (data <= upper_bound)
        result = np.where(within_3sigma, "pass", "not pass")
        adata.obs["cell_size_score"] = result
        print("The data is Gaussian (normal distribution) (fail to reject H0)")
    else:
        adata.obs["cell_size_score"] = np.ones(adata.n_obs)*(-1)
        print("The data is not Gaussian (reject H0)")

def cal_sensitivity_saturation(adata):
    filtered_genes = [gene for gene in adata.var_names if not gene.startswith("Negative") and not gene.startswith("SystemControl")]
    filtered_gene_data = adata[:, filtered_genes].X
        # Check if the data is a numpy array or a sparse matrix
    if isinstance(filtered_gene_data, np.ndarray):
        # For numpy array, count non-zero entries in each row (per cell)
        non_zero_columns_per_cell = np.count_nonzero(filtered_gene_data != 0, axis=1)
    else:
        # For sparse matrix, count non-zero entries in each row (per cell)
        non_zero_columns_per_cell = np.diff(filtered_gene_data.indptr)

    # Now, `non_zero_columns_per_cell` is a vector with the number of non-zero columns per cell
    trans = non_zero_columns_per_cell
    trans = trans.reshape((adata.n_obs, 1))
    umi = filtered_gene_data.sum(axis=1)
    umi = np.asarray(umi)
    Area = adata.obs["Area.um2"]
    Area = Area.to_numpy().reshape((adata.n_obs, 1))
    trans_total =  len(filtered_genes)


    adata.obs["sensitivity_1"] = trans/Area
    adata.obs["sensitivity_2"] = trans
    adata.obs["sensitivity_3"] = trans/(trans_total*Area)

    adata.obs["saturation_1"] = umi/Area
    adata.obs["saturation_2"] = umi
 

def cal_solidity_circularity(adata):
    
    polygon_df = adata.uns["polygon"]
    groups = polygon_df.groupby('fov_cell')
    output_dict = {}
    output_dict["name"] = []
    output_dict["solidity"] = []
    output_dict["circularity"] = []
    flag = 0
    
    for name, group in groups:
        output_dict["name"].append(name)
        polygon_coords = [(group.loc[i, "x_local_px"], group.loc[i, "y_local_px"]) for i in group.index]
        if len(polygon_coords) < 3:
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
    # Reindex the metadata to align with adata.obs_names
    # This ensures that the metadata aligns with the observations in the AnnData object
    out_df = out_df.reindex(adata.obs_names)
    adata.obs["solidity"] = out_df["solidity"]
    adata.obs["circularity"] = out_df["circularity"]
    



def check_normal_distribution(row):
    stat, p_value = shapiro(row)
    return p_value > 0.05  

def process_matrix(matrix):
    num_outliers = 0
    num_normal_rows = 0
    
    for row in matrix:
        original_row = row.copy()  
        if not check_normal_distribution(row):
            row = np.log(row + 1)  
            if not check_normal_distribution(row):
                continue  
        
        num_normal_rows += 1
        mean = np.mean(row)
        std = np.std(row)
        outliers = np.sum(np.abs(row - mean) > 3 * std)
        num_outliers += outliers

    if num_normal_rows > 0:
        return num_outliers / num_normal_rows
    else:
        return 0

def gft_r(signal, U):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    return U.T @ signal

def cal_gcc(knee, signal1, signal2, eigenvectors):
    signal1_data = signal1.toarray()
    signal2_data = signal2.toarray()

    gft_signal1 = gft_r(signal1_data, eigenvectors)
    gft_signal2 = gft_r(signal2_data, eigenvectors)
    similarity = 1 - cosine(gft_signal1[1:knee].flatten(), gft_signal2[1:knee].flatten())
    return similarity

def get_score_list(gene_df, polygon_df, cell_name, bin_size = 25, ratio_neighbors = 1, eigenvectors = None, knee = 0, box_extend_ratio = 0):
    
    tmp_ratio_neighbors = ratio_neighbors
    result_list = []
    # Record the start time
    start_time = time.time()


    # Group the data by 'fov_cell' (each group represents a polygon)
    groups = polygon_df.groupby('fov_cell')

    # Set the specific idx (fov_cell) that you want to select
    # specific_fov_cell = "12_126"  # Replace this with your desired cell index
    specific_fov_cell = cell_name

    # Filter the DataFrame to get the specific polygon by 'fov_cell'
    specific_polygon_group = polygon_df[polygon_df['fov_cell'] == specific_fov_cell]
    # Extract x and y coordinates for the current polygon
    x_coords = specific_polygon_group['x_local_px']
    y_coords = specific_polygon_group['y_local_px']

    # Append the first point to close the polygon
    x_coords = list(x_coords) + [x_coords.iloc[0]]
    y_coords = list(y_coords) + [y_coords.iloc[0]]

    # Create a shapely Polygon object from the coordinates
    polygon = Polygon(zip(x_coords, y_coords))
    
    print(f"This is polygon: {polygon}")

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

    filtered_sparse = csr_matrix(df)

    # # filtered_sparse_2 = csr_matrix(filtered_sparse_2)
    # # #count_mtx = adata[idx, :].X
    tmp_adata = ad.AnnData(filtered_sparse)
    
    #pd.DataFrame(tmp_adata.X.toarray()).to_csv("output_matrix.csv", index=False)



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
        
        # print(eigenvalues)
        knee = low_freq

    ########################################

    # print("Start preparing gene")
    # Filter and draw the genes that fall within the bounding box of the red polygon
    genes_in_box = gene_df[
        (gene_df['x_local_px'] >= minx0) &
        (gene_df['x_local_px'] <= maxx0) &
        (gene_df['y_local_px'] >= miny0) &
        (gene_df['y_local_px'] <= maxy0)
    ]

    # Classify genes as inside or outside the polygon
    point_list = []
    genes_in_box["x_modified_px"] = (genes_in_box['x_local_px'] - minx0)/((maxx0-minx0)/bin_size)
    genes_in_box["y_modified_px"] = (genes_in_box['y_local_px'] - miny0)/((maxy0-miny0)/bin_size)

    # Loop through each gene and check if it is inside or outside the red polygon
    for idx, gene in genes_in_box.iterrows():
        gene_point = Point(gene["x_modified_px"], gene["y_modified_px"])
        if gene["fov_cell"] == cell_name:
            point_list.append(gene_point)
        elif gene["CellComp"] not in ["Membrane","Nuclear","Cytoplasm"]:
            point_list.append(gene_point)


    # Convert the shapely Polygon to a list of shapes (as geo-rasterize expects)
    shapes = point_list
    
    # Foreground value for the red polygon (e.g., 1 for cells marked in red)
    foregrounds = [1 for i in range(len(point_list))]

    # Set the size of the raster grid (50x50)
    raster_size = (bin_size, bin_size)

    # Rasterize the polygon
    rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='add')
    # Turned to replace in binarized gene distribution
    # rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='replace')
    flattened_array = rasterized.flatten()

    # Step 2: Convert to a pandas DataFrame with a single column named 'mask'
    df = pd.DataFrame(flattened_array, columns=["mask"])
    filtered_sparse = csr_matrix(df)
    tmp_adata2 = ad.AnnData(filtered_sparse)
    tmp_adata2.obsm['array_row']  = x_global_px_array
    tmp_adata2.obsm["array_col"] = y_global_px_array
    
    cos_similarity = cal_gcc(knee, tmp_adata.X, tmp_adata2.X, eigenvectors)
    
    # print(f"This is knee: {knee}")
    # print(f"This is cos_similarity: {cos_similarity}")
    return cos_similarity, eigenvectors, knee

def cal_snr(knee_low, knee_high, low_signal, high_signal, eigenvectors_low, eigenvectors_high):
    low_signal_data = low_signal.toarray()
    high_signal_data = high_signal.toarray()

    gft_signal1 = gft_r(low_signal_data, eigenvectors_low)
    gft_signal2 = gft_r(high_signal_data, eigenvectors_high)
    
    # print("$$$$$$$$$$$$$$$$$$$$$$$")
    # print(f"This is low frequency signal: {low_signal}")
    # print("$$$$$$$$$$$$$$$$$$$$$$$")
    # print(f"This is high frequency signal: {high_signal}")
    
    snr = 10*np.log10((np.sum(gft_signal1[1:knee_low] ** 2))/(np.sum(gft_signal2[knee_high:-1] ** 2)))
    return snr

def get_score_list_trans(gene_df, gene_name, x_coords, y_coords, bin_size = 25, ratio_neighbors = 1, eigenvectors_low = None, knee_low = 0, eigenvectors_high = None, knee_high = 0):
    
    tmp_ratio_neighbors = ratio_neighbors
    # Append the first point to close the polygon
    x_coords = list(x_coords) + [x_coords[0]]
    y_coords = list(y_coords) + [y_coords[0]]

    # Create a shapely Polygon object from the coordinates
    polygon = Polygon(zip(x_coords, y_coords))
    print(f"Name of gene: {gene_name}")
    minx0, miny0, maxx0, maxy0 = polygon.bounds
    
    # print(f"This is minx0, miny0, maxx0, maxy0: {minx0, miny0, maxx0, maxy0}")
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

    filtered_sparse = csr_matrix(df)

    # # filtered_sparse_2 = csr_matrix(filtered_sparse_2)
    # # #count_mtx = adata[idx, :].X
    tmp_adata = ad.AnnData(filtered_sparse)
    
    #pd.DataFrame(tmp_adata.X.toarray()).to_csv("output_matrix.csv", index=False)

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
        low_freq, high_freq = gft_5.determine_frequency_ratio(tmp_adata, ratio_neighbors = tmp_ratio_neighbors)
        eigenvectors_low = tmp_adata.uns['FMs_after_select']["low_FMs"]
        eigenvalues_low = tmp_adata.uns['FMs_after_select']['low_FMs_frequency']
        
        eigenvectors_high = tmp_adata.uns['FMs_after_select']["high_FMs"]
        eigenvalues_high = tmp_adata.uns['FMs_after_select']['high_FMs_frequency']
        
        
        print("$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"This is low frequency engenvalues: {eigenvalues_low}, which size is: {eigenvectors_low.shape}")
        print("$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"This is high frequency engenvalues: {eigenvalues_high}, which size is: {eigenvectors_high.shape}")        
        
        # print(high_freq)
        knee_low = low_freq
        knee_high = high_freq

    ########################################
    
    point_list = []
    genes_in_box = gene_df
    genes_in_box["x_modified_px"] = (genes_in_box['x_local_px'] - minx0)/((maxx0-minx0)/bin_size)
    genes_in_box["y_modified_px"] = (genes_in_box['y_local_px'] - miny0)/((maxy0-miny0)/bin_size)
    # Loop through each gene and check if it is inside or outside the red polygon
    for idx, gene in genes_in_box.iterrows():
        gene_point = Point(gene["x_modified_px"], gene["y_modified_px"])
        if gene["target"] == gene_name:
            point_list.append(gene_point)

    # Convert the shapely Polygon to a list of shapes (as geo-rasterize expects)
    shapes = point_list
    
    # Foreground value for the red polygon (e.g., 1 for cells marked in red)
    foregrounds = [1 for i in range(len(point_list))]

    # Set the size of the raster grid (50x50)
    raster_size = (bin_size, bin_size)

    # Rasterize the polygon
    rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='add')
    # Turned to replace in binarized gene distribution
    # rasterized = rasterize(shapes, foregrounds, raster_size, algorithm='replace')
    flattened_array = rasterized.flatten()

    # Step 2: Convert to a pandas DataFrame with a single column named 'mask'
    df = pd.DataFrame(flattened_array, columns=["mask"])
    filtered_sparse = csr_matrix(df)
    tmp_adata2 = ad.AnnData(filtered_sparse)
    tmp_adata2.obsm['array_row']  = x_global_px_array
    tmp_adata2.obsm["array_col"] = y_global_px_array
    
    snr = cal_snr(knee_low, knee_high, tmp_adata2.X, tmp_adata2.X, eigenvectors_low, eigenvectors_high)
    
    print(f"This is knee: {knee_low}")
    print(f"This is knee: {knee_high}")
    print(f"This is snr: {snr}")
    return snr, eigenvectors_low, eigenvectors_high, knee_low, knee_high
    
def cal_sgcc(adata, bin_size = 60, ratio_neighbors = 10, box_extend_ratio = 0):
    
    tmp_bin_size = bin_size
    tmp_ratio_neighbors = ratio_neighbors
    
    gene_df = adata.uns["tx"]
    polygon_df = adata.uns["polygon"]
    
    groups = polygon_df.groupby('fov_cell')
    output_dict = {}
        
    output_dict["name"] = []
    output_dict["score"] = []
    flag = 0
    for name, group in groups:
        output_dict["name"].append(name)

        print(f"We are calculating sgcc, bs = {tmp_bin_size}, k = {tmp_ratio_neighbors}")
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
    out_df = out_df.reindex(adata.obs_names)
    adata.obs["sgcc"] = out_df["score"]

def cal_signal_noise_ratio(adata, bin_size = 60, ratio_neighbors = 10):
    
    tmp_bin_size = bin_size
    tmp_ratio_neighbors = ratio_neighbors
    
    gene_df = adata.uns["tx"]
    groups =  gene_df.groupby('target')
    x_coords = np.array([gene_df['x_local_px'].min(),gene_df['x_local_px'].min(),gene_df['x_local_px'].max(),gene_df['x_local_px'].max()])
    y_coords = np.array([gene_df['y_local_px'].min(),gene_df['y_local_px'].max(),gene_df['y_local_px'].max(),gene_df['x_local_px'].min()])
    
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
    out_df = out_df.reindex(adata.var_names)
    adata.var["snr"] = out_df["score"]

common_path = "/fs/ess/PAS1475/Xiaojie/spatialQC/test_data"
#for i in range(1,44,1):
for j in range(1):
    i = 12
   
    exprMat_file = os.path.join(common_path,f"sampled_exprMat_fov{i}.csv")
    metadata_file = os.path.join(common_path,f"sampled_metadata_fov{i}.csv")
    header = pd.read_csv(exprMat_file, nrows=1).columns.tolist()
    to_remove = ['fov', 'cell_ID']
    for item in to_remove:
            header.remove(item)
    header = [col for col in header if not col.startswith(('Negative', 'SystemControl'))]
    # Read the rest of the file, skipping the first row
    #df = pd.read_csv('sampled_exprMat2.csv', skiprows=1, header=None)
    df = pd.read_csv(exprMat_file)
    df['fov_cell'] = '12_' + df['cell_ID'].astype(str)
    df2 = df.drop(columns=['fov', 'cell_ID','fov_cell'])
    #filtered out the Negative and Systemcontrol columns
    df3 = df2.loc[:, ~df2.columns.str.startswith(('Negative', 'SystemControl'))]
    counts = csr_matrix(df3)
    adata = ad.AnnData(counts)

    # Set row name and column names
    adata.obs_names = df['fov_cell']
    adata.var_names = header

    count_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    print(count_df.head())

    #get metadata
    meta_df = pd.read_csv(metadata_file)
    meta_df = meta_df.set_index('fov_cell')
    # Reindex the metadata to align with adata.obs_names
    # This ensures that the metadata aligns with the observations in the AnnData object
    meta_df = meta_df.reindex(adata.obs_names)
    # Assign the metadata to adata.obs
    adata.obs = meta_df
    
    polygon_file_name = f"sampled_polygons_fov{i}.csv"
    tx_file_name = f"sampled_tx_fov{i}.csv"
    polygon_file_dir = os.path.join(common_path, polygon_file_name)
    tx_file_dir = os.path.join(common_path, tx_file_name)
    polygon_df = pd.read_csv(polygon_file_dir)
    polygon_df['fov_cell'] = '12_' + polygon_df['cellID'].astype(str)
    gene_df = pd.read_csv(tx_file_dir)
    gene_df['fov_cell'] = '12_' + gene_df['cell_ID'].astype(str)
    
    adata.uns["tx"] = gene_df
    adata.uns["polygon"] = polygon_df

    cal_EOR(adata)
    cal_cell_size(adata)
    cal_sensitivity_saturation(adata)
    cal_sgcc(adata)
    cal_signal_noise_ratio(adata)
    cal_solidity_circularity(adata)

    ###########################################################################################
    # calculate the probability of being doublet
    if isinstance(adata.X, np.ndarray):
        sparse_matrix = sp.csr_matrix(adata.X)
        csc_matrix = sparse_matrix.tocsc()
    else:
        csc_matrix = adata.X.tocsc()

    counts_matrix = csc_matrix
    # print("Matrix shape:", counts_matrix.shape)
    #Method 1: scrublet
    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=0.2, sim_doublet_ratio=5)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()
    result_1 = np.where(predicted_doublets, "doublet", "not doublet")
    adata.obs["scrublet_1"] = result_1


    out_put_file = os.path.join(common_path, f"obs_result_fov{i}.csv")
    out_put_file2 = os.path.join(common_path, f"var_result_fov{i}.csv")

    adata.obs.to_csv(out_put_file, index=True)
    adata.var.to_csv(out_put_file2, index=True)

    print(f"Successfully saved adata.obs to: {out_put_file}")
