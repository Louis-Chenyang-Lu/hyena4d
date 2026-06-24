#!/bin/bash
## smoke_test.sbatch: Slurm batch script for a quick smoke test of the training pipeline on a small subset of the data.
###########################################################################################################################
#SBATCH --time=01:00:00                        ## smoke test: 1 hour cap (secondary max is 4h)
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                      ## match num-workers=7 + 1 main process
#SBATCH --gres=gpu:V100:1                      ## same hardware as real run
#SBATCH --mem=8G
##–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#SBATCH --job-name="hyena_smoke"
#SBATCH --output=output_smoke/%x-%j.out
#SBATCH --error=output_smoke/%x-%j.err
#SBATCH --account=stat
#SBATCH --partition=secondary,stat,statstudent ## try secondary first (faster scheduling, 4h cap is fine for a smoke test)
#SBATCH --mail-type=ALL
#SBATCH --mail-user="lomasov2@illinois.edu"
###########################################################################################################################

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate hyena4d_linux

# Ensure output dir exists before slurm tries to write .out/.err on a fresh checkout.
# (If you submit from a clean directory, also run `mkdir -p output_smoke` on the login
# node before `sbatch smoke_test.sbatch`.)
mkdir -p output_smoke

python3 train_smoke_test.py