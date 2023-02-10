# Fine tune

./distributed_train.sh 4 /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_small_patch16_384.augreg_in1k --pretrained  --img-size 384 --sched cosine --epochs 8 --lr 1e-2 --batch-size 128 --workers 16 --clip-grad 1 --weight-decay 0.0 --warmup-epochs 0 --diff-res-ckpt 'output/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

# --experiment folder_name

# small resolution trained from scratch
./distributed_train.sh 4 /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_tiny_patch16_64  --img-size 64 --epochs 4 --lr 1e-2  --workers 16

python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_tiny_patch16_384 --pretrained --img-size 384

#   --diff-res-ckpt