#!/bin/bash

#SBATCH --account=hyuny
#SBATCH --job-name=PPO      ## Name of the job
#SBATCH --time=24:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8      ## The number of threads the code will use
#SBATCH --mem-per-cpu=2G     ## Real memory(MB) per CPU required by the job.
ARG=${1:-1}
LOGFILE="zztrainPPO_${ARG}_${SLURM_JOB_ID}.log"
exec >"$LOGFILE" 2>&1              # send all output (stdout+stderr) to the log

## not using the default python
module purge

## Execute the python script and pass the argument/input '90'
source ~/miniconda3/bin/activate hatchery
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

srun python PPO_pprdyn1_HPC.py "$ARG"