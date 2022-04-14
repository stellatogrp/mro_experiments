#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --array=10,20,30,40,50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4-00:00
#SBATCH -o /scratch/gpfs/iywang/mro_code/portfolio/results/portfolio_test_%A_N%a.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export GRB_LICENSE_FILE=/usr/licensed/gurobi/license/gurobi.lic

module purge
module load anaconda3
conda activate mroenv

python portfolio/testing.py --sparsity $SLURM_ARRAY_TASK_ID

#
#
#
#
#!/bin/zsh
#SBATCH -c 1
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:volta:1 -p gpu
#SBATCH -o /home/gridsan/stellato/results/online/portfolio/portfolio_test_%A_N%a.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bartolomeo.stellato@gmail.com
#
#
# # Activate environment
# . "/home/gridsan/stellato/miniconda/etc/profile.d/conda.sh"
# conda activate online
#
# # module load gurobi/8.0.1
# export GRB_LICENSE_FILE="/home/software/gurobi/gurobi.lic"
#
#
# # Run actual script
# HDF5_USE_FILE_LOCKING=FALSE python online_optimization/portfolio/testing.py --sparsity $SLURM_ARRAY_TASK_ID
