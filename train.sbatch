#!/bin/bash

# #SBATCH --job-name=PEP-T

#SBATCH --partition=day
#SBATCH --time=1-00:00:00

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60G
# the job will need 240GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:4
#the job can use and see 4 GPUs (4 GPUs are available in total on one node)



#SBATCH --error=logs/train.err.%J
#SBATCH --output=logs/train.out.%J

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=fabian.schreier@student.uni-tuebingen.de

error=0

if [ -z "$1" ]
then
    config=split1
    echo "Using config \"$config\": default"
else
    config=$1
    echo "Using config \"$config\": provided as \"$1\""
fi

touch ~/logs/train.gpu.$SLURM_JOB_ID


mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets
mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets/baseline
mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets/transfer
mkdir /scratch/$SLURM_JOB_ID/Output

echo Copying dataset
#cp -a ~/ProcessedDatasets/Cat2000.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz
cp -a ~/ProcessedDatasets/Cat2000_baseline.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000_baseline.tar.gz && error=1
cp -a ~/ProcessedDatasets/SaliCon_baseline.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon_baseline.tar.gz && error=1
cp -a ~/ProcessedDatasets/Mit1003_baseline.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Mit1003_baseline.tar.gz && error=1
cp -a ~/ProcessedDatasets/Cat2000_transfer.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000_transfer.tar.gz && error=1
cp -a ~/ProcessedDatasets/SaliCon_transfer.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon_transfer.tar.gz && error=1
cp -a ~/ProcessedDatasets/Mit1003_transfer.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Mit1003_transfer.tar.gz && error=1


echo Extracting dataset

tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000_baseline.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/baseline/ && error=1
tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon_baseline.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/baseline/ && error=1
tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/Mit1003_baseline.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/baseline/ && error=1
tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000_transfer.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/transfer/ && error=1
tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/SaliCon_transfer.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/transfer/ && error=1
tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/Mit1003_transfer.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/transfer/ && error=1


echo Copying source code
cp -a ~/src/ /scratch/$SLURM_JOB_ID/src

cp -a ~/TrainOutput/baseline_model.h5 /scratch/$SLURM_JOB_ID/baseline_model_input.h5

cp ~/gpu_util.sh /scratch/$SLURM_JOB_ID/gpu_util.sh
touch /scratch/$SLURM_JOB_ID/running

/scratch/$SLURM_JOB_ID/gpu_util.sh /scratch/$SLURM_JOB_ID/running ~/logs/train.gpu.$SLURM_JOB_ID &
gpu_util_pid=$!
echo Started GPU util as $gpu_util_pid


eval checkpoint_path=~/TrainOutput/checkpoints/baseline_model.$config-$SLURM_JOB_ID.{epoch:02d}.h5


echo Executing processing script
singularity exec ~/PerEng.1_15.simg python3 -u /scratch/$SLURM_JOB_ID/src/train.py \
    --config=$config \
    --checkpoint_file=$checkpoint_path \
     && error=1

if [ -f /scratch/$SLURM_JOB_ID/Output/baseline_model.h5 ];
then
    echo Copying output to home
    cp -af /scratch/$SLURM_JOB_ID/Output/baseline_model.h5 ~/TrainOutput/baseline_model.$config-$SLURM_JOB_ID.h5
    cp -af /scratch/$SLURM_JOB_ID/Output/history.pckl ~/TrainOutput/history.$config-$SLURM_JOB_ID.pckl
else
    echo "First train script failed"
fi

echo Stopping GPU util running as process $gpu_util_pid
rm /scratch/$SLURM_JOB_ID/running
kill -- -${gpu_util_pid}


echo Cleaning up job directory
rm -R /scratch/$SLURM_JOB_ID/ProcessedDatasets
rm -R /scratch/$SLURM_JOB_ID/Output



#if [ -f /scratch/$SLURM_JOB_ID/Output/baseline_model.h5 ];
#then
#    echo Moving log files to success folder
#    mv ~/logs/train.out.$SLURM_JOB_ID ~/logs/finished/train.out.$SLURM_JOB_ID
#    mv ~/logs/train.err.$SLURM_JOB_ID ~/logs/finished/train.err.$SLURM_JOB_ID
#    mv ~/logs/train.gpu.$SLURM_JOB_ID ~/logs/finished/train.gpu.$SLURM_JOB_ID
#else
#    echo Moving log files to error folder
#    mv ~/logs/train.out.$SLURM_JOB_ID ~/logs/errors/train.out.$SLURM_JOB_ID
#    mv ~/logs/train.err.$SLURM_JOB_ID ~/logs/errors/train.err.$SLURM_JOB_ID
#    mv ~/logs/train.gpu.$SLURM_JOB_ID ~/logs/errors/train.gpu.$SLURM_JOB_ID
#fi