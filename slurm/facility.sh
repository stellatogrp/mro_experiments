#!/bin/bash
#SBATCH --job-name=facilitytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4:00:00
#SBATCH -o /scratch/gpfs/iywang/mro_results/facility/cvar/facility_test_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

export MOSEKLM_LICENSE_FILE=/scratch/gpfs/iywang/mro_experiments/mosek/mosek.lic

module purge
module load anaconda3/2022.5
conda activate mroenv

python facility/facility.py --foldername /scratch/gpfs/iywang/mro_results/facility/cvar/

# python facility/plots.py --foldername /scratch/gpfs/iywang/mro_results/facility/cvar/


