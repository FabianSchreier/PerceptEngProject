#!/bin/bash


#SBATCH --job-name=PercEngProj-TestTrain

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=50G
# the job will need 100 of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:4
#the job can use and see 4 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=0:15:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=test.err.%J
#SBATCH --output=test.out.%J

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=fabian.schreier@student.uni-tuebingen.de


mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets
mkdir /scratch/$SLURM_JOB_ID/Output
echo Copying source code
cp -a ~/src/ /scratch/$SLURM_JOB_ID/src

echo Executing processing script
singularity exec ~/PerEng.simg python3 -u /scratch/$SLURM_JOB_ID/src/mnist_test.py

echo DONE!
