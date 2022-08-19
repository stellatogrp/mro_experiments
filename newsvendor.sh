#!/bin/bash
#SBATCH --job-name=newsvendortest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --time=60:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/newsvendor/MIP/test/newsvendor_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3 
conda activate mroenv

python newsvendor/MIP/newsMIP.py --foldername /scratch/gpfs/iywang/mro_results/newsvendor/MIP/m40_K500_r20

