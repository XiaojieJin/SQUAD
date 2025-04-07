# import h5py

# with h5py.File("/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/merfish_official/fov0_processed/cell_boundaries/feature_data_0.hdf5", "r") as f:
#     cell_id = "110883424764611924400221639916314253469"  # 你要查看的 cell ID
#     z1_group = f["featuredata"][cell_id]["zIndex_4"]["p_0"]["coordinates"]
#     print(list(z1_group))

# import json
# with open('/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/Xenium_data/gene_panel.json', 'r') as f:
#     data = json.load(f)
#     # 查找negative probe相关信息
#     print(data)

# import h5py

# f = h5py.File('/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/Xenium_data/cell_feature_matrix.h5', 'r')
# features = f['matrix']['features']['name'][:]
# negative_features = [f.decode('utf-8') for f in features if b'negative' in f.lower()]
# print('Negative features:', negative_features)

import pandas as pd

# 替换成你的实际路径
df = pd.read_parquet("/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/visiumhd_data/binned_outputs/square_008um/spatial/tissue_positions.parquet")

# 显示前几行
print(df.head())
            