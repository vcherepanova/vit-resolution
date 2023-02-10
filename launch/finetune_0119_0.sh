#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=vit-finetune                           # sets the job name if not set from environment
#SBATCH --array=0                                 # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output slurm-logs/%x_%A_%a.log                # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm-logs/%x_%A_%a.log                 # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=24:00:00                                 # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                             # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                 # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem 128gb                                      # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE         # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,

interpolation=bicubic
experiment=fine_tune-${interpolation}
FILE=output/train/${experiment}/last.pth.tar

if [ -f ${FILE} ];
then  
    ./distributed_train.sh 4 /fs/cml-datasets/ImageNet/ILSVRC2012 \
    --model vit_small_patch16_384.augreg_in1k \
    --pretrained  \
    --img-size 384 \
    --sched cosine \
    --epochs 8 \
    --lr 1e-2 \
    --batch-size 128 \
    --workers 16 \
    --clip-grad 1 \
    --weight-decay 0.0 \
    --warmup-epochs 0 \
    --diff-res-ckpt output/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz \
    --diff-res-interpolation ${interpolation} \
    --experiment ${experiment} \
    --resume ${FILE}
else
    ./distributed_train.sh 4 /fs/cml-datasets/ImageNet/ILSVRC2012 \
    --model vit_small_patch16_384.augreg_in1k \
    --pretrained  \
    --img-size 384 \
    --sched cosine \
    --epochs 8 \
    --lr 1e-2 \
    --batch-size 128 \
    --workers 16 \
    --clip-grad 1 \
    --weight-decay 0.0 \
    --warmup-epochs 0 \
    --diff-res-ckpt output/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz \
    --diff-res-interpolation ${interpolation} \
    --experiment ${experiment}
fi 