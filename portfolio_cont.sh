#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --array=10,20,30,40,50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4-00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio/results/portfolio_test_%A_N%a.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

module purge
module load anaconda3 
module load gurobi/9.5.1
conda activate mroenv

python portfolio/testing.py --sparsity $SLURM_ARRAY_TASK_ID

