#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from squad import Squad

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        # 设置数据路径
        data_path = "/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/RCC_TMA542_section05_v132_spatialdata"
        output_path = "/fs/ess/PAS1475/Xiaojie/spatialQC/test_data"
        
        # 检查数据目录是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据目录不存在: {data_path}")
        
        logger.info(f"开始处理Xenium数据: {data_path}")
        
        #创建Squad对象并读取数据
        squad = Squad.from_cosmx(
            common_path=data_path,
            fov_number=12,
            file_prefix = "RCC_BPC_23_tma542",
            # bin_size = 20, 
            # ratio_neighbors = 2, 
            # extend_ratio = 0.2
        )
        # squad = Squad.from_merscope(
        #     path=data_path,
        #     fov_number=0
        # )
        # squad = Squad.from_xenium(
        #     path=data_path,
        # )
        # squad = Squad.from_visiumhd(
        #     path = data_path,
        #     dataset_id = "Visium_HD_Human_Colon_Cancer_P1"
        # )

        # # 1. 输出unique的region值
        # logger.info("输出unique的region值...")
        # unique_regions = squad.sdata.points["transcripts"]["fov_name"].unique()
        # logger.info(f"Unique regions: {unique_regions}")
        # # 1. 输出unique的region值
        # logger.info("输出unique的region值...")
        # unique_regions = squad.sdata.table.obs["region"].unique()
        # logger.info(f"Unique regions: {unique_regions}")
        
        # # 2. 保存基因名到txt文件
        # logger.info("保存基因名到文件...")
        # gene_names_file = os.path.join(output_path, "gene_names.txt")
        # with open(gene_names_file, 'w') as f:
        #     for gene in squad.sdata.table.var_names:
        #         f.write(f"{gene}\n")
        # logger.info(f"基因名已保存到: {gene_names_file}")
        
        # # 3. 保存transcripts信息到csv文件
        # logger.info("保存transcripts信息...")
        # transcripts_df = squad.sdata.points["transcripts"]
        # transcripts_file = os.path.join(output_path, "transcripts_info.csv")
        # # 保存列名和前5行
        # transcripts_sample = transcripts_df.head()
        # transcripts_sample.to_csv(transcripts_file)
        # logger.info(f"Transcripts列名: {transcripts_df.columns.tolist()}")
        # logger.info(f"Transcripts信息已保存到: {transcripts_file}")
        
        # 计算所有质量控制指标
        logger.info("开始计算质量控制指标...")
        squad.compute_all_metrics()
        
        # 计算质量得分
        logger.info("开始计算质量得分...")
        squad.calculate_quality_score(
            output_prefix="xenium_quality",
            prob_threshold=0.05
        )


        # 保存最终结果
        output_prefix="merfish_quality"
        output_file = os.path.join(output_path, f"{output_prefix}_results.csv")
        logger.info(f"正在保存结果到: {output_file}")
        squad.sdata.table.obs.to_csv(output_file)
        
        logger.info("分析完成！")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
