
# the HPARAMS are mostly copied from discussion here: https://github.com/rwightman/pytorch-image-models/discussions/777
# with a few modifications according to hyper parameters defined here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/vit_i1k.py

HPARAMS="--sched cosine --epochs 10 --opt sgd -j 8 --warmup-epochs 1 --model-ema --model-ema-decay 0.99996 --remode pixel --reprob 0.25 --amp --weight-decay 0.1 --drop 0.0 --drop-path 0.0 --aa rand-m15-n2-mstd0.5-inc1 --mixup 0.2 --clip-grad 1.0 --lr 0.03 -b 512"
HPARAMS=${HPARAMS/'--epochs 10'/'--epochs 300'} # replace config to training from scratch config
HPARAMS=${HPARAMS/'--warmup-epochs 1'/'--warmup-epochs 33'} # 10000 steps = 10000*4096/1281167~32 epochs
HPARAMS=${HPARAMS/'--model-ema --model-ema-decay 0.99996'/''} # remove ema
HPARAMS=${HPARAMS/'--remode pixel --reprob 0.25'/'--remode pixel --reprob 0.25'} # keeping pixel erasure
HPARAMS=${HPARAMS/'--aa rand-m15-n2-mstd0.5-inc1'/'--aa rand-m15-n2-mmax30'} # modifying augmentation policy to according to Ross's suggestion
HPARAMS=${HPARAMS/'--mixup 0.2'/'--mixup 0.5'} # increasing mixup strength to match with the big vision implementation
HPARAMS=${HPARAMS/'--lr 0.03'/'--lr 0.001'} # modifying learning rate to be for training from scratch
HPARAMS=${HPARAMS/'--weight-decay 0.1'/'--weight-decay 0.1'} # weight decay in timm is 1000x larger than the big vision weight decay due to implementation differences


# ./distributed_train.sh 8 /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_tiny_patch16_64  --img-size 64 $HPARAMS
# ./distributed_train.sh 8 /fs/cml-datasets/ImageNet/ILSVRC2012 --model vitpeg_tiny_patch16_64  --img-size 64 $HPARAMS  
