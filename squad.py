import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel
from spatialdata.transformations import Scale, Translation, Sequence
import scrublet as scr
from utils import (
    cal_EOR, cal_cell_size, cal_sensitivity_saturation,
    cal_solidity_circularity, cal_sgcc, cal_signal_noise_ratio,
    calculate_quality_score, get_eigenvector, cal_scrublet
)
import glob
import shutil
import h5py

# 配置日志
logger = logging.getLogger(__name__)

def validate_spatialdata(sdata):
    """验证 SpatialData 对象是否包含必要的字段
    
    Args:
        sdata (SpatialData): 要验证的SpatialData对象
        
    Returns:
        bool: 验证是否通过
        
    Raises:
        ValueError: 当缺少必要字段时
    """
    required_fields = ['table', 'shapes', 'points']
    for field in required_fields:
        if not hasattr(sdata, field):
            raise ValueError(f"SpatialData对象缺少必要字段: {field}")
    return True
class Squad:
    default_bin_size = 60
    default_ratio_neighbors = 10
    default_extend_ratio = 0.1
    def __init__(self):
        self.sdata = None
        self.metadata = {}
        # self.bin_size = 60
        # self.ratio_neighbors = 10
        # self.extend_ratio = 0.1
        logger.info("初始化Squad对象")
    @classmethod
    def from_cosmx(cls, fov_number, common_path, file_prefix, bin_size = None, ratio_neighbors = None, extend_ratio = None):
        """从CSV文件创建Squad对象
        
        Args:
            fov_number (int): FOV编号
            common_path (str): 数据文件所在路径
            file_prefix (str): 文件前缀
            
        Returns:
            Squad: 新创建的Squad对象
            
        Raises:
            Exception: 当数据加载失败时
        """
        try:
            logger.info(f"正在从CSV文件创建Squad对象 - FOV: {fov_number}")
            self = cls()
            self.bin_size = bin_size if bin_size is not None else cls.default_bin_size
            self.ratio_neighbors = ratio_neighbors if ratio_neighbors is not None else cls.default_ratio_neighbors
            self.extend_ratio = extend_ratio if extend_ratio is not None else cls.default_extend_ratio

            # 检查目录是否存在
            if not os.path.exists(common_path):
                raise FileNotFoundError(f"目录不存在: {common_path}")
            
            output_dir = os.path.join(common_path, f"fov{fov_number}_processed")
            if os.path.exists(output_dir):
                logger.info("process文件夹存在")
            else:
                # 创建FOV特定的输出目录
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"创建输出目录: {output_dir}")

                # 处理CellLabels和CellComposite文件夹
                for folder in ['CellLabels', 'CellComposite']:
                    src_folder = os.path.join(common_path, folder)
                    if os.path.exists(src_folder):
                        logger.info(f"处理文件夹: {folder}")
                        if int(fov_number) < 10:
                            file_pattern = os.path.join(src_folder, f"*F0000{fov_number}.tif")
                        elif int(fov_number) < 100:
                            file_pattern = os.path.join(src_folder, f"*F000{fov_number}.tif")
                        else:
                            file_pattern = os.path.join(src_folder, f"*F00{fov_number}.tif")
                        logger.info(f"查找文件: {file_pattern}")
                        files = glob.glob(file_pattern)
                        
                        if not files:
                            logger.warning(f"在 {folder} 中没有找到FOV {fov_number}的文件")
                            continue
                            
                        dst_folder = os.path.join(output_dir, folder)
                        os.makedirs(dst_folder, exist_ok=True)
                        
                        # 复制对应FOV的文件
                        for file in files:
                            logger.info(f"复制文件: {file} -> {os.path.join(dst_folder, os.path.basename(file))}")
                            shutil.copy2(file, os.path.join(dst_folder, os.path.basename(file)))
                
                # 处理CSV文件
                for file in os.listdir(common_path):
                    if file.endswith('.csv'):
                        src_file = os.path.join(common_path, file)
                        dst_file = os.path.join(output_dir, file)
                        logger.info(f"处理CSV文件: {file}")
                        
                        # 读取CSV文件并筛选FOV
                        df = pd.read_csv(src_file)
                        if "metadata_file" in file:
                            df = df.drop(columns=["cell_id"])
                        else:
                            df.columns = df.columns.str.lower()
                            if 'cellid' in df.columns:
                                df.rename(columns={'cellid': 'cell_id'}, inplace=True)
                            # 如果列中有 'cell_id'，改名为 'cell_ID'
                            df.rename(columns={"cell_id": "cell_ID"}, inplace=True)

                        if ('fov' in df.columns):
                            df = df[df['fov'] == fov_number]
                            logger.info(f"筛选后的数据形状: {df.shape}")
                            df.to_csv(dst_file, index=False)
                            logger.info(f"保存到: {dst_file}")
                        else:
                            logger.warning(f"文件 {file} 中没有fov列")
                            # 如果没有fov列，直接复制文件
                            shutil.copy2(src_file, dst_file)
            
            # 使用处理后的目录读取数据
            logger.info(f"正在读取处理后的数据: {output_dir}")
            from spatialdata_io import cosmx
            self.sdata = cosmx(
                path=output_dir,
                dataset_id=file_prefix,
                transcripts=True
            )
            logger.info(f"FOV {fov_number}的数据读取成功")
            
            if not self.sdata.shapes:
                polygon_file = glob.glob(os.path.join(common_path, f"fov{fov_number}_processed/*polygons.csv"))
                tmp_df = pd.read_csv(polygon_file[0])
                tmp_df.columns = tmp_df.columns.str.lower()
                if 'cellid' in tmp_df.columns:
                    tmp_df.rename(columns={'cellid': 'cell_id'}, inplace=True)
                # 如果列中有 'cell_id'，改名为 'cell_ID'
                tmp_df.rename(columns={"cell_id": "cell_ID"}, inplace=True)
                if tmp_df["fov"].nunique() > 1:
                    tmp_df = tmp_df[tmp_df['fov'] == fov_number]
                self.metadata["polygon_file"] = tmp_df
                
            self.metadata["fov_number"] = fov_number
            self.metadata["path"] = common_path
            logger.info("Squad对象创建成功")
            logger.info("数据读取成功:")
            logger.info(f"数据形状: {self.sdata.table.shape}")
            logger.info(f"基因数量: {self.sdata.table.n_vars}")
            logger.info(f"细胞数量: {self.sdata.table.n_obs}")
            return self
        except Exception as e:
            logger.error(f"创建Squad对象失败: {str(e)}")
            raise

    @classmethod
    def from_merscope(cls, bin_size = None, ratio_neighbors = None, extend_ratio = None, fov_number=None, path=None,):
        """从MERSCOPE数据创建Squad对象

        Args:
            path (str): MERSCOPE数据目录路径
            fov_number (int, optional): FOV编号

        Returns:
            Squad: 新创建的Squad对象

        Raises:
            Exception: 当数据加载失败时
        """
        try:
            logger.info(f"正在从MERSCOPE数据创建Squad对象 - 路径: {path}")
            self = cls()
            self.bin_size = bin_size if bin_size is not None else cls.default_bin_size
            self.ratio_neighbors = ratio_neighbors if ratio_neighbors is not None else cls.default_ratio_neighbors
            self.extend_ratio = extend_ratio if extend_ratio is not None else cls.default_extend_ratio

            # 创建FOV特定的输出目录
            output_dir = path if fov_number is None else os.path.join(path, f"fov{fov_number}_processed")
            if os.path.exists(output_dir):
                logger.info("process文件夹存在")
            else:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"输出目录准备完毕: {output_dir}")

                # 处理 cell_boundaries 文件夹
                for folder in ['cell_boundaries']:
                    src_folder = os.path.join(path, folder)
                    if os.path.exists(src_folder):
                        logger.info(f"处理文件夹: {folder}")
                        file_pattern = os.path.join(src_folder, f"*_{fov_number}.hdf5")
                        logger.info(f"查找文件: {file_pattern}")
                        files = glob.glob(file_pattern)

                        if not files:
                            logger.warning(f"在 {folder} 中没有找到FOV {fov_number}的文件")
                            continue

                        dst_folder = os.path.join(output_dir, folder)
                        os.makedirs(dst_folder, exist_ok=True)

                        for file in files:
                            logger.info(f"复制文件: {file} -> {os.path.join(dst_folder, os.path.basename(file))}")
                            shutil.copy2(file, os.path.join(dst_folder, os.path.basename(file)))

                # 添加 Images 文件夹的软链接（避免复制大文件）
                images_src = os.path.join(path, "images")
                images_dst = os.path.join(output_dir, "images")

                if os.path.exists(images_src) and not os.path.exists(images_dst):
                    os.symlink(images_src, images_dst)
                    logger.info(f"创建 Images 的软链接: {images_dst} -> {images_src}")
                else:
                    logger.info("Images 文件夹软链接已存在或原始路径不存在，跳过")

                # 处理 CSV 文件
                for file in ['detected_transcripts.csv', 'cell_metadata.csv']:
                    src_file = os.path.join(path, file)
                    if os.path.exists(src_file):
                        dst_file = os.path.join(output_dir, file)
                        logger.info(f"处理CSV文件: {file}")

                        df = pd.read_csv(src_file, index_col=0)
                        if fov_number is not None and 'fov' in df.columns:
                            df = df[df['fov'] == fov_number]
                            logger.info(f"筛选后的数据形状: {df.shape}")
                        df.to_csv(dst_file)
                        logger.info(f"保存到: {dst_file}")

                # 对齐 cell_by_gene.csv 和 cell_metadata.csv 的索引
                count_path = os.path.join(path, "cell_by_gene.csv")
                count_path_new = os.path.join(output_dir, "cell_by_gene.csv")
                obs_path = os.path.join(output_dir, "cell_metadata.csv")

                count_df = pd.read_csv(count_path, index_col=0, dtype=str)
                obs_df = pd.read_csv(obs_path, index_col=0, dtype=str)

                common_index = count_df.index.intersection(obs_df.index)
                if len(common_index) == 0:
                    raise ValueError("没有公共细胞ID，检查两个文件的index是否命名一致")

                count_df_aligned = count_df.loc[common_index].sort_index()
                obs_df_aligned = obs_df.loc[common_index].sort_index()

                count_df_aligned.to_csv(count_path_new)
                obs_df_aligned.to_csv(obs_path)

            # 使用 spatialdata_io 读取数据
            from spatialdata_io import merscope
            logger.info("开始读取MERSCOPE数据...")
            self.sdata = merscope(
                path=output_dir,
                transcripts=True
            )
            print(f"If not self.sdata.shapes: {not self.sdata.shapes}")
            all_coords = [] 
            if not self.sdata.shapes:
                h5py_dir = os.path.join(path, f"cell_boundaries/feature_data_{fov_number}.hdf5")        
                with h5py.File(h5py_dir, "r") as f:
                    for cell_id in f["featuredata"].keys():
                        z1_group = f["featuredata"][cell_id]["zIndex_4"]
                        if "p_0" in list(z1_group.keys()):
                            coords = z1_group["p_0"]["coordinates"]
                            # 如果你有多个 cell，循环处理每个 cell 的坐标即可
                            coords = coords[0]  # 取出 shape 为 (67, 2)

                            # 拆成 x/y 两列
                            df = pd.DataFrame(coords, columns=["x", "y"])
                            df.insert(0, "cell_ID", cell_id)  # 插入 cell_ID 列到最前
                            all_coords.append(df)
                final_df = pd.concat(all_coords, ignore_index=True)
                self.metadata["polygon_file"] = final_df
            #     tmp_df = pd.read_csv(polygon_file)
            #     tmp_df.columns = tmp_df.columns.str.lower()
            #     if 'cellid' in tmp_df.columns:
            #         tmp_df.rename(columns={'cellid': 'cell_id'}, inplace=True)
            #     # 如果列中有 'cell_id'，改名为 'cell_ID'
            #     tmp_df.rename(columns={"cell_id": "cell_ID"}, inplace=True)
            #     if tmp_df["fov"].nunique() > 1:
            #         tmp_df = tmp_df[tmp_df['fov'] == fov_number]
            #     self.metadata["polygon_file"] = tmp_df

            self.metadata["path"] = path
            if fov_number is not None:
                self.metadata["fov_number"] = fov_number

            logger.info("数据读取成功:")
            logger.info(f"数据形状: {self.sdata.table.shape}")
            logger.info(f"基因数量: {self.sdata.table.n_vars}")
            logger.info(f"细胞数量: {self.sdata.table.n_obs}")

            return self

        except Exception as e:
            logger.error(f"创建Squad对象失败: {str(e)}")
            raise

    @classmethod
    def from_xenium(cls, path, bin_size = None, ratio_neighbors = None, extend_ratio = None):
        """从CSV文件创建Squad对象
        
        Args:
            fov_number (int): FOV编号
            common_path (str): 数据文件所在路径
            file_prefix (str): 文件前缀
            
        Returns:
            Squad: 新创建的Squad对象
            
        Raises:
            Exception: 当数据加载失败时
        """
        try:
            logger.info(f"正在从CSV文件创建Squad对象")
            self = cls()
            self.bin_size = bin_size if bin_size is not None else cls.default_bin_size
            self.ratio_neighbors = ratio_neighbors if ratio_neighbors is not None else cls.default_ratio_neighbors
            self.extend_ratio = extend_ratio if extend_ratio is not None else cls.default_extend_ratio

            from spatialdata_io import xenium

            # 加载 Xenium 数据（路径为解压后的 outs 文件夹）
            self.sdata = xenium(
                path,
                cells_as_circles=True,       # 用圆形简化 cell 表示，可加速可视化
                cells_labels=True,           # 加载 cell segmentation label（推荐用于分析）
                nucleus_labels=True,         # 加载 nucleus segmentation label（如果存在）
                cells_boundaries=True,       # 加载多边形 cell 边界（便于可视化）
                nucleus_boundaries=True,     # 加载 nucleus 边界（如需）
                transcripts=True,            # 加载单分子 RNA 空间坐标
                morphology_focus=True,       # 加载 morphology 图像
                morphology_mip=True,         # 加载 morphology MIP 图像（v2.0 以下）
                aligned_images=True          # 自动加载 H&E/IF 对齐图像（如果有）
            )
            self.metadata["polygon_file"] = None
            self.metadata["fov_number"] = None
            self.metadata["path"] = path
            logger.info("Squad对象创建成功")
            logger.info("数据读取成功:")
            logger.info(f"数据形状: {self.sdata.table.shape}")
            logger.info(f"基因数量: {self.sdata.table.n_vars}")
            logger.info(f"细胞数量: {self.sdata.table.n_obs}")
            return self
        except Exception as e:
            logger.error(f"创建Squad对象失败: {str(e)}")
            raise        
    
    @classmethod
    def from_visiumhd(cls, path, dataset_id, bin_size = None, ratio_neighbors = None, extend_ratio = None):
        """从VisiumHD数据创建Squad对象
        
        Args:
            path (str): VisiumHD数据目录路径
        """
        try:
            logger.info(f"正在从VisiumHD数据创建Squad对象")
            self = cls()
            self.bin_size = bin_size if bin_size is not None else cls.default_bin_size
            self.ratio_neighbors = ratio_neighbors if ratio_neighbors is not None else cls.default_ratio_neighbors
            self.extend_ratio = extend_ratio if extend_ratio is not None else cls.default_extend_ratio

            from spatialdata_io import visium_hd

            sdata = visium_hd(
                path=path,
                dataset_id=dataset_id
            )
            self.sdata = sdata
            self.sdata.table = self.sdata.tables['square_008um']
            self.metadata["polygon_file"] = None
            self.metadata["fov_number"] = None
            self.metadata["path"] = path
            logger.info("Squad对象创建成功")
            logger.info("数据读取成功:")
            logger.info(f"数据形状: {self.sdata.table.shape}")
            logger.info(f"基因数量: {self.sdata.table.n_vars}")
            logger.info(f"细胞数量: {self.sdata.table.n_obs}")
            return self
        except Exception as e:
            logger.error(f"创建Squad对象失败: {str(e)}")
            raise    
            

    def compute_all_metrics(self):
        """计算所有质量控制指标
        
        Raises:
            ValueError: 当SpatialData对象无效时
            Exception: 当计算过程中出错时
        """
        try:
            logger.info("开始计算质量控制指标")
            validate_spatialdata(self.sdata)
            
            logger.info("计算EOR...")
            cal_EOR(self.sdata, polygon_file=self.metadata["polygon_file"])
            
            logger.info("计算细胞大小...")
            cal_cell_size(self.sdata)
            
            logger.info("计算敏感度和饱和度...")
            cal_sensitivity_saturation(self.sdata, polygon_file=self.metadata["polygon_file"])

            logger.info("计算实心度和圆形度...")
            cal_solidity_circularity(self.sdata, polygon_file=self.metadata["polygon_file"]) #TBD

            logger.info("计算双胞体得分...")
            cal_scrublet(self.sdata)

            if not "array_row" in self.sdata.table.obs.columns:
                logger.info("计算Fourier Mode")
                eigenvectors_low, eigenvectors_high, knee_low, knee_high = get_eigenvector(bin_size = self.bin_size, neighbors_ratio = self.ratio_neighbors)
                logger.info("计算空间基因共表达系数...")
                cal_sgcc(self.sdata, polygon_file=self.metadata["polygon_file"], bin_size = self.bin_size, ratio_neighbors = self.ratio_neighbors, box_extend_ratio = self.extend_ratio, eigenvectors_low = eigenvectors_low, knee_low = knee_low) #TBD
                
                logger.info("计算信噪比...")
                cal_signal_noise_ratio(self.sdata, polygon_file=self.metadata["polygon_file"], bin_size = self.bin_size, ratio_neighbors = self.ratio_neighbors, eigenvectors_low = eigenvectors_low, knee_low = knee_low, eigenvectors_high = eigenvectors_high, knee_high = knee_high) #TBD
            
            logger.info("所有质量控制指标计算完成")
        except Exception as e:
            logger.error(f"计算质量控制指标时出错: {str(e)}")
            raise

    def calculate_quality_score(self, output_prefix="quality_output", prob_threshold=0.05):
        """计算质量得分
        
        Args:
            output_prefix (str): 输出文件前缀
            prob_threshold (float): 概率阈值
            
        Raises:
            ValueError: 当SpatialData对象无效时
            Exception: 当计算过程中出错时
        """
        try:
            logger.info(f"开始计算质量得分 - 输出前缀: {output_prefix}")
            validate_spatialdata(self.sdata)
            if not "array_row" in self.sdata.table.obs.columns:
                feature_columns = [
                    'EOR', 'cell_size_score', 'sensitivity_2',
                    'saturation_2', 'solidity', 'circularity', 'sgcc', 
                    'scrublet_1'
                ]
            else:
                feature_columns = [
                    'EOR', 'cell_size_score', 'sensitivity_2',
                    'saturation_2', 'solidity', 'circularity']
            # feature_columns = [
            #     'EOR', 'cell_size_score', 'sensitivity_1', 'sensitivity_2', 'sensitivity_3',
            #     'saturation_1', 'saturation_2', 'solidity', 'circularity'
            # ]
            calculate_quality_score(
                self.sdata, 
                feature_columns, 
                output_prefix=output_prefix, 
                prob_threshold=prob_threshold
            )
            logger.info("质量得分计算完成")
        except Exception as e:
            logger.error(f"计算质量得分时出错: {str(e)}")
            raise