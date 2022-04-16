#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --array=10,20,30,40,50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --time=40:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio/MIP/m=100,K=1000,r=10/portfolio_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_code/mosek/mosek.lic

module purge
module load anaconda3 
#module load gurobi/9.5.1
conda activate mroenv

python portfolio/portMIP_testing.py 
#--sparsity $SLURM_ARRAY_TASK_ID

