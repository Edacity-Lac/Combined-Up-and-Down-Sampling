#!/bin/bash
# best=True, checkpoint="./checkpoint/model_99.pth" 表示在model_90.pth到model_100.pth这10个权重中选择最好的性能
# best=False, checkpoint="./checkpoint/model_99.pth" 表示仅测试model_99.pth这个权重
conda activate segnet && \
python test.py \
    --model SegNet \
    --dataset camvid \
    --num_workers 4 \
    --batch_size 16 \
    --best  \
    --checkpoint "./checkpoint/camvid/SegNetbs8gpu1_train/model_199.pth" \
    --cuda True \
    # --gpus \
    # --save_seg_dir \
    # --save \