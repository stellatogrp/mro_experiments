#!/bin/bash
#SBATCH --job-name=newstest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/newsvendor/news_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2022.5
conda activate mroenv

python newsvendor/newsvendor.py --foldername /scratch/gpfs/iywang/mro_results/newsvendor/

# python newsvendor/newsvendor_outlier.py --foldername /scratch/gpfs/iywang/mro_results/newsvendor/

# python newsvendor/newsvendor_support.py --foldername /scratch/gpfs/iywang/mro_results/newsvendor/

# python newsvendor/plots.py --foldername /scratch/gpfs/iywang/mro_results/newsvendor/

# python newsvendor/plots_compare.py --foldername /scratch/gpfs/iywang/mro_results/newsvendor/