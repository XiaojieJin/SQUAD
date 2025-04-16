import pandas as pd

# 读取两个文件
edges_1 = pd.read_csv("/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/adjacency_edges.csv")
edges_2 = pd.read_csv("/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/12_adjacency_edges.csv")

# 标准化边顺序（无向边）
for df in [edges_1, edges_2]:
    df.columns = ["source", "target", "weight"]
    df["min_node"] = df[["source", "target"]].min(axis=1)
    df["max_node"] = df[["source", "target"]].max(axis=1)
    df["edge"] = list(zip(df["min_node"], df["max_node"]))

edges_1_cleaned = edges_1[["edge", "weight"]].sort_values(by=["edge", "weight"]).reset_index(drop=True)
edges_2_cleaned = edges_2[["edge", "weight"]].sort_values(by=["edge", "weight"]).reset_index(drop=True)

# 比较是否完全一致
are_equal = edges_1_cleaned.equals(edges_2_cleaned)
print("完全一致:", are_equal)

# 如果不一致，打印不同之处
if not are_equal:
    merged = pd.merge(edges_1_cleaned, edges_2_cleaned, how='outer', on=["edge", "weight"], indicator=True)
    diff = merged[merged["_merge"] != "both"]
    print("差异条目数:", len(diff))
    print(diff.head())