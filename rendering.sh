#!/bin/bash
iters="300000_head.tar"
names="cnn2_25fps"
datasets="cnn2_25fps"
near=0.5555068731307984
far=1.1555068731307983
path="dataset/finetune_models/${names}/${iters}"
datapath="dataset/${datasets}/0"
bc_type="torso_imgs"
suffix="val"
python NeRFs/run_nerf_deform.py --need_torso True --config dataset/test_config.txt --expname ${names}_${suffix} --expname_finetune ${names}_${suffix} --render_only --ft_path ${path} --datadir ${datapath} --bc_type ${bc_type} --near ${near} --far ${far}

