import sys
import os
from pathlib import Path
from pyqupath.geojson import mask_to_geojson
import tifffile
import argparse
import glob
from tifffile import TiffFile

gating_dict = {"1_CD3": {"min": 200, "max": 2000}, "1_CD8": {"min": 200, "max": 2000}}
# # 设置deepcell token
# with open("/bmbl_data/xiaojie/Spatial_QC/Indepth_data_processing/deepcell_token.bashrc", "r") as f:
#     token = f.read().strip()
# os.environ.update({"DEEPCELL_ACCESS_TOKEN": token})

# 添加pycodex包所在目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pycodex import segmentation

def apply_gating_to_ometiff(ome_tiff_path, output_path, gating_dict, channel_names):
    import tifffile
    import numpy as np
    from pathlib import Path

    ome = tifffile.imread(ome_tiff_path)
    with tifffile.TiffFile(ome_tiff_path) as tif:
        metadata = tif.ome_metadata

    adjusted = []
    for idx, name in enumerate(channel_names):
        img = ome[idx]
        if name in gating_dict:
            gmin = gating_dict[name]["min"]
            gmax = gating_dict[name]["max"]
            img = np.clip(img, gmin, gmax)
        adjusted.append(img.astype(np.uint16))  # 确保统一类型

    adjusted_array = np.stack(adjusted)
    tifffile.imwrite(output_path, adjusted_array, description=metadata)

def extract_channel_names(tiff_path):
    with TiffFile(tiff_path) as tif:
        ome_metadata = tif.ome_metadata

    # 用 xml 解析器解析 channel 名称
    import xml.etree.ElementTree as ET
    root = ET.fromstring(ome_metadata)
    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    channels = root.findall(".//ome:Channel", namespaces=namespace)
    channel_names = [c.attrib['Name'] for c in channels]
    return channel_names

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--segmentation_results', type=str, default='segmentation_results_0', help='Segmentation results directory')
    parser.add_argument('--internal_markers', nargs='+', default=["1_DAPI"], help='List of internal markers')
    parser.add_argument('--boundary_markers', nargs='+', default=["1_CD45RO","1_CD3","1_CD8"], help='List of boundary markers')
    parser.add_argument('--maxima_threshold', type=float, default=0.075, help='Maxima threshold')
    parser.add_argument('--interior_threshold', type=float, default=0.20, help='Interior threshold')
    return parser.parse_args()

args = parse_args()

# 设置输入输出路径
data_dir = Path(args.data_dir)
#ometiff_path = data_dir / "combined_images.ome.tiff"
# output_dir = data_dir / "segmentation_results"
# output_dir.mkdir(exist_ok=True)
ome_tiff_path = glob.glob(os.path.join(data_dir, "*.ome.tiff"))[0]
# ome = tifffile.imread(ome_tiff_path)  #
apply_gating_to_ometiff(
    ome_tiff_path=ome_tiff_path,
    output_path=data_dir / "combined_images_gated.ome.tiff",
    gating_dict=gating_dict,
    channel_names=extract_channel_names(ome_tiff_path)
)
ome_tiff_path_new = data_dir / "combined_images_gated.ome.tiff"
# 设置标记物
internal_markers = args.internal_markers
boundary_markers = args.boundary_markers

# 设置参数
pixel_size_um = 0.5068164319979996  # CosMx的像素大小
maxima_threshold = args.maxima_threshold
interior_threshold = args.interior_threshold

# 运行分割
segmentation.run_segmentation_mesmer_cell(
    unit_dir=data_dir,
    internal_markers=internal_markers,
    boundary_markers=boundary_markers,
    thresh_q_min=0.0,  # 最小阈值分位数
    thresh_q_max=0.99,  # 最大阈值分位数
    thresh_otsu=False,  # 使用OTSU阈值
    scale=True,  # 缩放图像
    pixel_size_um=pixel_size_um,
    maxima_threshold=maxima_threshold,
    interior_threshold=interior_threshold,
    ometiff_path=ome_tiff_path_new,
    tag=args.segmentation_results
)
segmentation_mask = tifffile.imread(data_dir / args.segmentation_results / "segmentation_mask.tiff")
mask_to_geojson(
    mask=segmentation_mask,
    geojson_path=data_dir / args.segmentation_results / "segmentation_mask.geojson",
)
print("分割完成！") 

# from tifffile import TiffFile

# with TiffFile("/bmbl_data/xiaojie/Spatial_QC/Indepth_data_processing_0417/fov_1.ome.tiff") as tif:
#     ome_metadata = tif.ome_metadata
#     print(ome_metadata)
