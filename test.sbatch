#!/bin/bash
# #SBATCH --job-name=PEP-Ts

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60G
# the job will need 240GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:4
#the job can use and see 4 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=24:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=logs/test.err.%J
#SBATCH --output=logs/test.out.%J

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=fabian.schreier@student.uni-tuebingen.de

error=0

if [ -z "$1" ]
then
    config=test
    echo "Using config \"$config\": default"
else
    config=$1
    echo "Using config \"$config\": provided as \"$1\""
fi


touch ~/logs/test.gpu.$SLURM_JOB_ID


mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets
mkdir /scratch/$SLURM_JOB_ID/Output

echo Copying datasets

#cp -a ~/ProcessedDatasets/Cat2000.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz
cp -a ~/ProcessedDatasets/Cat2000_small_output.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz && error=1
cp -a ~/ProcessedDatasets/SaliCon_small_output.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon.tar.gz && error=1

echo Extracting datasets

tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/ && error=1
tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/ && error=1

ls -al /scratch/$SLURM_JOB_ID/ProcessedDatasets/

echo Copying source code
cp -a ~/src/ /scratch/$SLURM_JOB_ID/src

cp ~/gpu_util.sh /scratch/$SLURM_JOB_ID/gpu_util.sh
touch /scratch/$SLURM_JOB_ID/running

/scratch/$SLURM_JOB_ID/gpu_util.sh /scratch/$SLURM_JOB_ID/running ~/logs/test.gpu.$SLURM_JOB_ID &
gpu_util_pid=$!
echo Started GPU util as $gpu_util_pid


eval checkpoint_path=~/TrainOutput/checkpoints/baseline_model.$config-$SLURM_JOB_ID.{epoch:02d}.h5

echo Executing processing script
singularity exec ~/PerEng.1_15.simg python3 -u /scratch/$SLURM_JOB_ID/src/train.py \
    --config=$config \
    --epochs=1 \
     && error=1

echo Stopping GPU util running as process $gpu_util_pid
rm /scratch/$SLURM_JOB_ID/running
kill -- -${gpu_util_pid}


echo Cleaning up job directory
rm -R /scratch/$SLURM_JOB_ID/ProcessedDatasets
rm -R /scratch/$SLURM_JOB_ID/Output

#if [ $error -eq 0 ];
#then
#    echo Moving log files to success folder
#    mv ~/logs/test.out.$SLURM_JOB_ID ~/logs/finished/test.out.$SLURM_JOB_ID
#    mv ~/logs/test.err.$SLURM_JOB_ID ~/logs/finished/test.err.$SLURM_JOB_ID
#    mv ~/logs/test.gpu.$SLURM_JOB_ID ~/logs/finished/test.gpu.$SLURM_JOB_ID
#else
#    echo Moving log files to error folder
#    mv ~/logs/test.out.$SLURM_JOB_ID ~/logs/errors/test.out.$SLURM_JOB_ID
#    mv ~/logs/test.err.$SLURM_JOB_ID ~/logs/errors/test.err.$SLURM_JOB_ID
#    mv ~/logs/test.gpu.$SLURM_JOB_ID ~/logs/errors/test.gpu.$SLURM_JOB_ID
#fi