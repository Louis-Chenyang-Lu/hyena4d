#!/bin/bash
###########################################################################################################################
#SBATCH --time=04:00:00                      ## total expected computing time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                      ## number of cores per task (if not specified, default is 1)
#SBATCH --gres=gpu:V100:1                      ## number of gpus per node, select between gpu:V100:1, gpu:V100:2, gpu:H100:1, gpu:H100:2 (if not specified, no gpu will be allocated)
#SBATCH --mem=32G                              ## total memory required(entire memory,if not specified)
##–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#SBATCH --job-name="hyena_pretrain"            ## specify a custom name for the job
#SBATCH --output=output/run1_secondary_ilia/%x-%j.out    ## specify the output file name pattern, %x for job name and %j for job id
#SBATCH --error=output/run1_secondary_ilia/%x-%j.err     ## specify the error file name pattern, %x for job name and %j for job id
#SBATCH --account=stat                         ## account to charge for the job
#SBATCH --partition=secondary       	       ## queue. select between stat, statstudent, secondary, scavenger (preemptible, not recommended for long jobs without checkpointing)
#SBATCH --mail-type=ALL                        ## (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user="lomasov2@illinois.edu"    ## email address to notify
###########################################################################################################################

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate hyena4d_linux

# make output directory if it doesn't exist
mkdir -p output/run1_secondary_ilia

python3 train_hyena.py \
    --input-dir input/pretrain/normalized \
    --output-dir output/run1_secondary_ilia \
    --subject-field 0 \
    --num-workers 7 \
    --patch-size 9 4 5 10 \
    --patch-chunk 20 \
    --t-sub 50 \
    --batch-size 32 \
    --epochs 50 \







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


# # (base) [lomasov2@cc-login5 ~]$ sinfo -p stat,statstudent -N -o "%.8N %.4c %.25P %.9m %.12l %.12L %G"
# # NODELIST CPUS                 PARTITION    MEMORY    TIMELIMIT  DEFAULTTIME GRES
# #  ccc0085   40                      stat     95900  14-00:00:00        30:00 (null)
# #  ccc0087   40                      stat     95900  14-00:00:00        30:00 (null)
# #  ccc0088   40                      stat     95900  14-00:00:00        30:00 (null)
# #  ccc0089   40               statstudent     96000   7-00:00:00        30:00 gpu:V100:2
# #  ccc0089   40                      stat     96000  14-00:00:00        30:00 gpu:V100:2
# #  ccc0090   40               statstudent     96000   7-00:00:00        30:00 gpu:V100:2
# #  ccc0090   40                      stat     96000  14-00:00:00        30:00 gpu:V100:2
# #  ccc0270   40               statstudent    386500   7-00:00:00        30:00 (null)
# #  ccc0270   40                      stat    386500  14-00:00:00        30:00 (null)
# #  ccc0271   40               statstudent    386500   7-00:00:00        30:00 (null)
# #  ccc0271   40                      stat    386500  14-00:00:00        30:00 (null)
# #  ccc0454   96               statstudent    257000   7-00:00:00        30:00 (null)
# #  ccc0454   96                      stat    257000  14-00:00:00        30:00 (null)
# #  ccc0455   96               statstudent    257000   7-00:00:00        30:00 (null)
# #  ccc0455   96                      stat    257000  14-00:00:00        30:00 (null)
# #  ccc0456   96               statstudent    257000   7-00:00:00        30:00 gpu:H100:1
# #  ccc0456   96                      stat    257000  14-00:00:00        30:00 gpu:H100:1
# #  ccc0457   96               statstudent    257000   7-00:00:00        30:00 gpu:H100:1
# #  ccc0457   96                      stat    257000  14-00:00:00        30:00 gpu:H100:1
# #  ccc0493   96                      stat    257000  14-00:00:00        30:00 (null)
# #  ccc0494   96                      stat    257000  14-00:00:00        30:00 (null)


# # (hyena4d_linux) [lomasov2@cc-login3 ~]$ sinfo -p secondary -N -o "%.8N %.4c %.25P %.9m %.12l %.12L %G"
# # NODELIST CPUS                 PARTITION    MEMORY    TIMELIMIT  DEFAULTTIME GRES
# #  ccc0207   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0208   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0209   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0210   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0211   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0212   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0213   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0214   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0215   40                 secondary    192960      4:00:00        10:00 gpu:V100:1
# #  ccc0218   40                 secondary    193000      4:00:00        10:00 (null)
# #  ccc0219   40                 secondary    193000      4:00:00        10:00 (null)
# #  ccc0223   40                 secondary    386490      4:00:00        10:00 (null)
# #  ccc0265   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0266   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0270   40                 secondary    386500      4:00:00        10:00 (null)
# #  ccc0271   40                 secondary    386500      4:00:00        10:00 (null)
# #  ccc0273   40                 secondary    192960      4:00:00        10:00 (null)
# #  ccc0274   40                 secondary    386500      4:00:00        10:00 (null)
# #  ccc0276   40                 secondary    193000      4:00:00        10:00 (null)
# #  ccc0277   40                 secondary    193000      4:00:00        10:00 (null)
# #  ccc0278   40                 secondary    193000      4:00:00        10:00 (null)
# #  ccc0279  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0286  128                 secondary    515580      4:00:00        10:00 gpu:TeslaT4:3
# #  ccc0287  128                 secondary    515580      4:00:00        10:00 gpu:TeslaT4:3
# #  ccc0288  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0289  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0290  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0291  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0292  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0293  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0294  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0295  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0296  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0297  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0298  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0303  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0308   64                 secondary    257490      4:00:00        10:00 (null)
# #  ccc0309   64                 secondary    257490      4:00:00        10:00 (null)
# #  ccc0310   64                 secondary    257490      4:00:00        10:00 (null)
# #  ccc0311   64                 secondary    257490      4:00:00        10:00 (null)
# #  ccc0324   16                 secondary   1019300      4:00:00        10:00 gpu:RTXA6000:2
# #  ccc0333   64                 secondary    257300      4:00:00        10:00 gpu:A40:3
# #  ccc0334   64                 secondary    257490      4:00:00        10:00 (null)
# #  ccc0338  128                 secondary    257530      4:00:00        10:00 (null)
# #  ccc0340  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0341  128                 secondary    257530      4:00:00        10:00 (null)
# #  ccc0347   64                 secondary   1031410      4:00:00        10:00 (null)
# #  ccc0348   64                 secondary    515315      4:00:00        10:00 (null)
# #  ccc0349   64                 secondary    515315      4:00:00        10:00 (null)
# #  ccc0350   64                 secondary    515315      4:00:00        10:00 (null)
# #  ccc0351   64                 secondary    515315      4:00:00        10:00 gpu:A100:4
# #  ccc0352  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0353  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0354  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0355  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0356  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0357  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0358  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0359  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0360  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0361  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0362  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0363  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0364  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0365  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0366  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0367  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0368  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0370   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0371   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0372   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0373   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0374   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0375   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0376   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0377   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0378   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0379   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0380   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0381   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0382   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0383   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0384   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0385   64                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0400  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0401  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0402  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0403  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0404  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0405  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0406  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0407  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0408  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0409  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0410  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0411  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0412  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0413  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0414  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0415  128                 secondary    257270      4:00:00        10:00 (null)
# #  ccc0417  128                 secondary    515500      4:00:00        10:00 (null)
# #  ccc0418   64                 secondary   4128000      4:00:00        10:00 (null)
# #  ccc0419   96                 secondary   1031000      4:00:00        10:00 gpu:H100:8
# #  ccc0420   96                 secondary   1031000      4:00:00        10:00 gpu:H100:8
# #  ccc0421   96                 secondary   1031000      4:00:00        10:00 gpu:H100:8
# #  ccc0422  192                 secondary   1031000      4:00:00        10:00 gpu:H100:8
# #  ccc0423   96                 secondary   1031000      4:00:00        10:00 gpu:H100:8
# #  ccc0424   96                 secondary   1031000      4:00:00        10:00 gpu:H100:8
# #  ccc0425   96                 secondary    515000      4:00:00        10:00 (null)
# #  ccc0426   96                 secondary   1031000      4:00:00        10:00 (null)
# #  ccc0427  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0428  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0429  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0430  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0431  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0432  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0433  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0434  128                 secondary   1031550      4:00:00        10:00 (null)
# #  ccc0437  128                 secondary   2051645      4:00:00        10:00 (null)
# #  ccc0443   96                 secondary    257400      4:00:00        10:00 (null)
# #  ccc0444   96                 secondary    257400      4:00:00        10:00 (null)
# #  ccc0445   96                 secondary    257400      4:00:00        10:00 (null)
# #  ccc0446   96                 secondary    257400      4:00:00        10:00 (null)
# #  ccc0454   96                 secondary    257000      4:00:00        10:00 (null)
# #  ccc0455   96                 secondary    257000      4:00:00        10:00 (null)
# #  ccc0456   96                 secondary    257000      4:00:00        10:00 gpu:H100:1
# #  ccc0457   96                 secondary    257000      4:00:00        10:00 gpu:H100:1
# #  ccc0459   64                 secondary    515000      4:00:00        10:00 (null)
# #  ccc0460   64                 secondary    515000      4:00:00        10:00 (null)
# #  ccc0461   64                 secondary    515000      4:00:00        10:00 (null)
# #  ccc0462   64                 secondary    515000      4:00:00        10:00 (null)
# #  ccc0463   64                 secondary    515000      4:00:00        10:00 (null)
# #  ccc0464   64                 secondary    515000      4:00:00        10:00 (null)
# #  ccc0469  128                 secondary   1031450      4:00:00        10:00 (null)
# #  ccc0478   96                 secondary    257250      4:00:00        10:00 gpu:A30:2
# #  ccc0493   96                 secondary    257000      4:00:00        10:00 (null)
# #  ccc0494   96                 secondary    257000      4:00:00        10:00 (null)
# #  ccc0510   48                 secondary    255398      4:00:00        10:00 (null)


# # (base) [lomasov2@cc-login5 ~]$ sreport -M icc -thour cluster AccountUtilizationByUser account=stat Start="2025-09-01"
# # --------------------------------------------------------------------------------
# # Cluster/Account/User Utilization 2025-09-01T00:00:00 - 2026-05-28T00:59:59 (23245200 secs)
# # Usage reported in CPU Hours
# # --------------------------------------------------------------------------------
# #   Cluster         Account     Login     Proper Name      Used   Energy 
# # --------- --------------- --------- --------------- --------- -------- 
# #       icc            stat                              259372        0 
# #       icc            stat  bleidig2          Leidig      7577        0 
# #       icc            stat      br36     Ravishankar         5        0 
# #       icc            stat  chenyuz9           Zhang      2254        0 
# #       icc            stat     ckhin            Khin      4070        0 
# #       icc            stat   echen53            Chen       593        0 
# #       icc            stat  eugeneh2             Han       147        0 
# #       icc            stat  jaideep3         Mahajan        15        0 
# #       icc            stat  rbrahma2          Brahma       485        0 
# #       icc            stat   sayanc3     Chakrabarty     65281        0 
# #       icc            stat    syun13             Yun         0        0 
# #       icc            stat  szhan105           Zhang      3669        0 
# #       icc            stat   yihanl7             Lin       384        0 
# #       icc            stat    yinan5           Cheng    174892        0 