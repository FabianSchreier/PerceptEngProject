#!/bin/bash


#SBATCH --job-name=PercEngProj-Train-Cat2000

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60G
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

touch ~/train.gpu.$SLURM_JOB_ID


mkdir /scratch/$SLURM_JOB_ID/ProcessedDatasets
mkdir /scratch/$SLURM_JOB_ID/Output

echo Copying dataset
#cp -a ~/ProcessedDatasets/Cat2000.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz
cp -a ~/ProcessedDatasets/Cat2000_small_output.tar.gz /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz


ls -al /scratch/$SLURM_JOB_ID/ProcessedDatasets/

echo Extracting dataset

tar -zxf /scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000.tar.gz -C /scratch/$SLURM_JOB_ID/ProcessedDatasets/

ls -al /scratch/$SLURM_JOB_ID/ProcessedDatasets/

echo Copying source code
cp -a ~/src/ /scratch/$SLURM_JOB_ID/src

cp -a ~/TrainOutput/baseline_model.e100.h5 /scratch/$SLURM_JOB_ID/baseline_model_input.h5

cp ~/gpu_util.sh /scratch/$SLURM_JOB_ID/gpu_util.sh
touch /scratch/$SLURM_JOB_ID/running

/scratch/$SLURM_JOB_ID/gpu_util.sh /scratch/$SLURM_JOB_ID/running ~/train.gpu.$SLURM_JOB_ID &

eval checkpoint_path=~/TrainOutput/checkpoints/baseline_model.e150.{epoch:02d}.h5

echo Executing second training
singularity exec ~/PerEng.1_15.simg python3 -u /scratch/$SLURM_JOB_ID/src/train.py \
    --dataset=/scratch/$SLURM_JOB_ID/ProcessedDatasets/Cat2000 \
    --output_folder=/scratch/$SLURM_JOB_ID/Output \
    --epochs=150 --initial_epoch=100 \
    --model_file=/scratch/$SLURM_JOB_ID/baseline_model_input.h5 --lr=0.0005 --recompile \
    --checkpoint_file=$checkpoint_path \
    --early_stop

if [ -f  /scratch/$SLURM_JOB_ID/Output/baseline_model.h5 ];
then
    echo Copying output to home
    cp -af /scratch/$SLURM_JOB_ID/Output/baseline_model.h5 ~/TrainOutput/baseline_model.e150.h5
    cp -af /scratch/$SLURM_JOB_ID/Output/history.pckl ~/TrainOutput/history.e150.pckl
else
    echo "Second train script failed"
fi

echo Stopping GPU util
rm /scratch/$SLURM_JOB_ID/running


echo Cleaning up job directory
rm -R /scratch/$SLURM_JOB_ID/ProcessedDatasets
rm -R /scratch/$SLURM_JOB_ID/Output

echo DONE!
