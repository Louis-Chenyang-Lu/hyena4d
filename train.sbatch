#!/bin/bash
###########################################################################################################################
#SBATCH --time=20:00:00                        ## total expected computing time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
##SBATCH --mem-per-cpu=8G                      ## total memory required(entire memory,if not specified)
##–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#SBATCH --job-name="hyena_pretrain"            ## specify a custom name for the job
#SBATCH --output=output/run1_ilia/%x-%j.out    ## specify the output file name pattern, %x for job name and %j for job id
#SBATCH --error=output/run1_ilia/%x-%j.err     ## specify the error file name pattern, %x for job name and %j for job id
#SBATCH --account=stat                         ## account to charge for the job
#SBATCH --partition=stat	                   ## queue. select between stat, statstudent, secondary, scavenger (preemptible, not recommended for long jobs without checkpointing)
#SBATCH --mail-type=ALL                        ## (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user="lomasov2@illinois.edu"    ## email address to notify
###########################################################################################################################

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate hyena4d_linux

# make output directory if it doesn't exist
mkdir -p output/run1_ilia

python3 train_hyena.py \
    --input-dir input/pretrain/normalized \
    --output-dir output/run1_ilia \
    --subject-field 0 \
    --patch-size 9 4 5 10 \
    --t-sub 50 \
    --batch-size 32 \
    --epochs 50







#### Additional information for Campus Cluster users:

# # (base) [lomasov2@cc-login1 ~]$ /sw/cc.users/tools/my.accounts

# # Campus Cluster user lomasov2's Account list:
# # Account=stat

# # (base) [lomasov2@cc-login1 ~]$ sinfo -s -o "%.25R %.12l %.12L %.5D"
# #                 PARTITION    TIMELIMIT  DEFAULTTIME NODES
# #                 secondary      4:00:00        10:00   138
# #                 scavenger   1-00:00:00        30:00    51
# #                      stat  14-00:00:00        30:00    13
# #               statstudent   7-00:00:00        30:00     8
# # (base) [lomasov2@cc-login1 ~]$ sinfo -p stat -N -o "%.8N %.4c %.25P %.9m %.12l %.12L %G"
# # NODELIST CPUS                 PARTITION    MEMORY    TIMELIMIT  DEFAULTTIME GRES
# #  ccc0085   40                      stat     95900  14-00:00:00        30:00 (null)
# #  ccc0087   40                      stat     95900  14-00:00:00        30:00 (null)
# #  ccc0088   40                      stat     95900  14-00:00:00        30:00 (null)
# #  ccc0089   40                      stat     96000  14-00:00:00        30:00 gpu:V100:2
# #  ccc0090   40                      stat     96000  14-00:00:00        30:00 gpu:V100:2
# #  ccc0270   40                      stat    386500  14-00:00:00        30:00 (null)
# #  ccc0271   40                      stat    386500  14-00:00:00        30:00 (null)
# #  ccc0454   96                      stat    257000  14-00:00:00        30:00 (null)
# #  ccc0455   96                      stat    257000  14-00:00:00        30:00 (null)
# #  ccc0456   96                      stat    257000  14-00:00:00        30:00 gpu:H100:1
# #  ccc0457   96                      stat    257000  14-00:00:00        30:00 gpu:H100:1
# #  ccc0493   96                      stat    257000  14-00:00:00        30:00 (null)
# #  ccc0494   96                      stat    257000  14-00:00:00        30:00 (null)
# # (base) [lomasov2@cc-login1 ~]$ sinfo -p statstudent -N -o "%.8N %.4c %.25P %.9m %.12l %.12L %G"
# # NODELIST CPUS                 PARTITION    MEMORY    TIMELIMIT  DEFAULTTIME GRES
# #  ccc0089   40               statstudent     96000   7-00:00:00        30:00 gpu:V100:2
# #  ccc0090   40               statstudent     96000   7-00:00:00        30:00 gpu:V100:2
# #  ccc0270   40               statstudent    386500   7-00:00:00        30:00 (null)
# #  ccc0271   40               statstudent    386500   7-00:00:00        30:00 (null)
# #  ccc0454   96               statstudent    257000   7-00:00:00        30:00 (null)
# #  ccc0455   96               statstudent    257000   7-00:00:00        30:00 (null)
# #  ccc0456   96               statstudent    257000   7-00:00:00        30:00 gpu:H100:1
# #  ccc0457   96               statstudent    257000   7-00:00:00        30:00 gpu:H100:1