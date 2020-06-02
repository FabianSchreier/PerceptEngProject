#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=PEP-P

#resources:

#SBATCH --cpus-per-task=24
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:0
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=24:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=logs/process.err.%J
#SBATCH --output=logs/process.out.%J

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=fabian.schreier@student.uni-tuebingen.de

error=0

if [ -z "$1" ]
then
    dataset=Cat2000
    echo "Using dataset \"$dataset\": default"
else
    dataset=$1
    echo "Using config \"$dataset\": provided as \"$1\""
fi


mkdir /scratch/$SLURM_JOB_ID/Datasets
mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets

ls -al /scratch/$SLURM_JOB_ID/Datasets

echo Copying dataset
cp -a ~/Datasets/$dataset /scratch/$SLURM_JOB_ID/Datasets/$dataset

ls -al /scratch/$SLURM_JOB_ID/Datasets

echo Copying source code
cp -a ~/src/ /scratch/$SLURM_JOB_ID/src

echo Executing processing script
singularity exec ~/PerEng.1_15.simg python3 -u /scratch/$SLURM_JOB_ID/src/process.py \
        --dataset=$dataset \
        --dataset_root=/scratch/$SLURM_JOB_ID/Datasets \
        --output_root=/scratch/$SLURM_JOB_ID/ProcessedDatasets \
        --parallel_entries=50

echo Archiving process dataset
tar -zcf /scratch/$SLURM_JOB_ID/ProcessedDatasets/${dataset}.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/ $dataset

ls -al /scratch/$SLURM_JOB_ID/ProcessedDatasets

echo Copying processed dataset ${dataset}.tar.gz to home ${dataset}_baseline.tar.gz
rm -R ~/ProcessedDatasets/${dataset}_baseline.tar.gz
cp -a /scratch/$SLURM_JOB_ID/ProcessedDatasets/${dataset}.tar.gz ~/ProcessedDatasets/${dataset}_baseline.tar.gz

echo Cleaning up job directory
rm -R /scratch/$SLURM_JOB_ID/Datasets
rm -R /scratch/$SLURM_JOB_ID/ProcessedDatasets

echo DONE!
