#!/bin/bash
#SBATCH --job-name=newsvendortest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=3G
#SBATCH --time=60:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/newsvendor/cont/test/newsvendor_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_code/mosek/mosek.lic

module purge
module load anaconda3 
#module load gurobi/9.5.1
conda activate mroenv

python newsvendor/newscont_testing.py 
#--sparsity $SLURM_ARRAY_TASK_ID

