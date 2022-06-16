#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -A quantum
#SBATCH --output=10-2-0-3-3.log
#SBATCH --partition=standard
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda

#command line format: numSamples, avgPhotons, displacment, maxPhotons, dataset to use
time python ../GBS-Kernel.py 10 2 0.0 3 3