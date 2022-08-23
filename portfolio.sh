#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --time=60:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/portfolio/MIP/m50_K300_r12/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3 
conda activate mroenv


#python portfolio/MIP/portMIP.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/MIP/m50_K300_r12/

python portfolio/MIP/plots.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/MIP/m50_K300_r12/


#python portfolio/cont/portcont.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/cont/m200_K900_r10/

#python portfolio/cont/plots.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/cont/m200_K900_r10/