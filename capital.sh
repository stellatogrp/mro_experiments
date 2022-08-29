#!/bin/bash
#SBATCH --job-name=capitaltest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=8G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/capital/m12_K50_r30/capital_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3 
conda activate mroenv

python capital_budgeting/capital1.py --foldername /scratch/gpfs/iywang/mro_results/capital/m12_K50_r30/

#python capital_budgeting/plots.py --foldername /scratch/gpfs/iywang/mro_results/capital/m12_K50_r30/

