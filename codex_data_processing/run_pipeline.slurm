#!/bin/bash
#SBATCH --account=pcon0022
#SBATCH --job-name=run_all_scripts
#SBATCH --mem=10000mb
#SBATCH --time=1:00:00

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"

module load miniconda3/24.1.2-py310
source activate python_env_3.9

# 运行02_run_segmentation.py
python 02_run_segmentation.py --data_dir data --segmentation_results segmentation_results_0 --internal_markers 1_DAPI --boundary_markers 1_CD45RO 1_CD3 --maxima_threshold 0.075 --interior_threshold 0.20

# 运行04_preprocessing.py
python 04_preprocessing.py --data_dir data --segmentation_results segmentation_results_0 --preprocessing_results preprocessing_results --markers 1_DAPI 1_CD3 1_CD4-150ms 1_FoxP3 1_CD8 1_CD68 3_CD206-250ms 1_CD163 1_CD11c

# 运行05_annotation.py
python 05_annotation.py --data_dir data --annotation_results annotation_results --segmentation_results segmentation_results_0 --preprocessing_results preprocessing_results --markers_phenotypic CD3 CD4 CD8 FoxP3 CD20 CD163 CD68 CD11c CD31 PDL1-250 HLA1 Ki67 BCL2 Pax5

# 运行06_plot_heatmap.py
python 06_plot_heatmap.py --data_dir data --annotation_results annotation_results --markers_phenotypic 1_CD3 1_CD4-150ms 1_FoxP3 1_CD8 1_CD68 3_CD206-250ms 1_CD163 1_CD11c

# 运行07_plot_umap.py
python 07_plot_umap.py --data_dir data --annotation_results annotation_results --markers_phenotypic 1_CD3 1_CD4-150ms 1_FoxP3 1_CD8 1_CD68 3_CD206-250ms 1_CD163 1_CD11c

echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"