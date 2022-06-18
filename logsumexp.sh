#!/bin/bash
#SBATCH --job-name=logtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=10
#SBATCH --time=40:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/logsumexp/m30_K90_r50/logtest_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_code/mosek/mosek.lic

module purge
module load anaconda3 
conda activate mroenv

python logsumexp/logsumexp_testing.py 

