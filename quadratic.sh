#!/bin/bash
#SBATCH --job-name=quadratictest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/quadratic_concave/m10_K60_r20/quadratic_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3 
conda activate mroenv

python quadratic_concave/quadratic.py --foldername /scratch/gpfs/iywang/mro_results/quadratic_concave/m10_K60_r20/

