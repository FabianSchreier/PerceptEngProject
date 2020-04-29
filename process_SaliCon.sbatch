#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=PercEngProj-Process-SaliCon

#resources:

#SBATCH --cpus-per-task=24
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:0
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=12:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=process.err.%J

#SBATCH --output=process.out.%J

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=fabian.schreier@student.uni-tuebingen.de
# your mail address


mkdir /scratch/$SLURM_JOB_ID/Datasets
mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets

echo Copying dataset
cp -a ~/Datasets/SaliCon /scratch/$SLURM_JOB_ID/Datasets/SaliCon

echo Copying source code
cp -a ~/src/ /scratch/$SLURM_JOB_ID/src

echo Executing processing script
singularity exec ~/PerEng.simg python3 -u /scratch/$SLURM_JOB_ID/src/process.py \
        --dataset_root=/scratch/$SLURM_JOB_ID/Datasets \
        --output_root=/scratch/$SLURM_JOB_ID/ProcessedDatasets \
        --parallel_entries=50

echo Archiving process dataset
tar -zcf /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/ SaliCon

echo Copying processed dataset to home
rm -R ~/ProcessedDatasets/SaliCon_small.tar.gz
cp -a /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon.tar.gz ~/ProcessedDatasets/SaliCon_small.tar.gz

echo Cleaning up job directory
rm -R /scratch/$SLURM_JOB_ID/Datasets
rm -R /scratch/$SLURM_JOB_ID/ProcessedDatasets

echo DONE!
