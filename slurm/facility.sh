#!/bin/bash
#SBATCH --job-name=facilitytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/facility/m25n5_K50_r15/facility_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2022.5
conda activate mroenv

python facility/facility.py --foldername /scratch/gpfs/iywang/mro_results/facility/m25n5_K50_r15/

# python facility/plots.py --foldername /scratch/gpfs/iywang/mro_results/facility/m25n5_K50_r15/

