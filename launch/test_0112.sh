./distributed_train.sh 2 /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_tiny_patch16_384 --pretrained  --img-size 384 --sched cosine --epochs 150 --lr 1e-2 --batch-size 64 --diff-res-ckpt 'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'

./distributed_train.sh 1 /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_tiny_patch16_384 --pretrained  --img-size 384 --sched cosine --epochs 150 --lr 1e-2 --batch-size 256 --diff-res-ckpt 'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz' --workers 4

python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_tiny_patch16_384 --pretrained --img-size 384

# --clip-grad 1 --batch-size 512 --weight-decay 0.0