#!/bin/bash
#SBATCH --job-name=facilitytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=5
#SBATCH --time=40:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/facility/m50n10_K100_r20_1/facility_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_code/mosek/mosek.lic

module purge
module load anaconda3 
#module load gurobi/9.5.1
conda activate mroenv

python facility/facilityseparate_testing1.py 

