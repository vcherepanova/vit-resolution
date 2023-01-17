python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch16_384 --pretrained  --img-size 384 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz --results_file results_interpolation_inference/results_base_16_384_int_pos_bilinear.csv --interpolation_res bilinear
python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch16_384 --pretrained  --img-size 384 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz --results_file results_interpolation_inference/results_base_16_384_int_pos_bicubic.csv --interpolation_res bicubic
python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch16_384 --pretrained  --img-size 384 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz --results_file results_interpolation_inference/results_base_16_384_int_pos_nearest.csv --interpolation_res nearest

python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch16_384 --pretrained  --img-size 384 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz --results_file results_interpolation_inference/results_base_16_384_baseline1.csv --interpolation_res bilinear
python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch16_224 --pretrained  --img-size 224 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz --results_file results_interpolation_inference/results_base_16_224_baseline2.csv --interpolation_res bilinear

python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch32_384 --pretrained  --img-size 384 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz --results_file results_interpolation_inference/results_base_32_384_int_patch_bilinear.csv --interpolation_res bilinear
python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch32_384 --pretrained  --img-size 384 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz --results_file results_interpolation_inference/results_base_32_384_int_patch_bicubic.csv --interpolation_res bicubic
python validate.py /fs/cml-datasets/ImageNet/ILSVRC2012 --model vit_base_patch32_384 --pretrained  --img-size 384 --diff-res-ckpt pretrained/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz --results_file results_interpolation_inference/results_base_32_384_int_patch_nearest.csv --interpolation_res nearest