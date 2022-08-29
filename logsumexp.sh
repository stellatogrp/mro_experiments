#!/bin/bash
#SBATCH --job-name=logtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/logsumexp/m50_K150_r20/logtest_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3 
conda activate mroenv

#python logsumexp/logsumexp.py --foldername /scratch/gpfs/iywang/mro_results/logsumexp/m50_K150_r20/

python logsumexp/plots.py --foldername /scratch/gpfs/iywang/mro_results/logsumexp/m50_K150_r20/