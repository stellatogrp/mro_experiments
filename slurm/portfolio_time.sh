#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio/K5_r10/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2024.2
conda activate mroenv

python portfolio/MIP/portMIP.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/new/K5_r10/

# python portfolio/MIP/plots.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/new/m30_K1000_r10/
