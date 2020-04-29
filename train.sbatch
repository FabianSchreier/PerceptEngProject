#!/bin/bash


#SBATCH --job-name=PercEngProj-Train-Cat2000

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24G
# the job will need 240GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:4
#the job can use and see 4 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=48:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=train.err.%J
#SBATCH --output=train.out.%J

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=fabian.schreier@student.uni-tuebingen.de


mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets
mkdir /scratch/$SLURM_JOB_ID/Output

echo Copying dataset
cp -a ~/ProcessedDatasets/Cat2000_small.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz

ls -al /scratch/$SLURM_JOB_ID/ProcessedDatasets/

echo Extracting dataset

tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/

ls -al /scratch/$SLURM_JOB_ID/ProcessedDatasets/

echo Copying source code
cp -a ~/src/ /scratch/$SLURM_JOB_ID/src

echo Executing processing script
singularity exec ~/PerEng.simg python3 -u /scratch/$SLURM_JOB_ID/src/train.py --dataset=/scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000 --output_folder=/scratch/$SLURM_JOB_ID/Output --epochs=50

echo Copying output to home
rm -R ~/TrainOutput
cp -a /scratch/$SLURM_JOB_ID/Output ~/TrainOutput

echo Cleaning up job directory
rm -R /scratch/$SLURM_JOB_ID/ProcessedDatasets
rm -R /scratch/$SLURM_JOB_ID/Output

echo DONE!
