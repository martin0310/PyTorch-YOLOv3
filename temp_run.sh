#!/bin/bash


# normal 1xN + block pattern prune 
# python -m pytorchyolo.train --data config/coco.data --pretrained_weights weights/yolov3.weights --block_pattern_prune --epochs 5


# resume 1xN + block pattern prune
# python -m pytorchyolo.train --data config/coco.data --pretrained_weights weights/yolov3.weights --block_pattern_prune --epochs 5 --resume_from /mnt/Data-Weight/1xN_new/yolov3/checkpoint/every_layer_pattern/4_patterns/yolov3_last.pth

# admm train
# python -m pytorchyolo.admm_train --data config/coco.data --pretrained_weights weights/yolov3.weights --epochs 5

# resume admm train
python -m pytorchyolo.admm_train --data config/coco.data --pretrained_weights weights/yolov3.weights --epochs 5 --resume_from /mnt/Data-Weight/1xN_new/yolov3/checkpoint/admm/4_patterns/admm_yolov3_last.pth

# retrain admm
# python -m pytorchyolo.train --data config/coco.data --pretrained_weights weights/yolov3.weights --admm_retrain --admm_checkpoint /mnt/Data-Weight/1xN_new/yolov3/checkpoint/admm/4_patterns/admm_yolov3_best_pruned.pth --epochs 5

# resume retrain admm 
# python -m pytorchyolo.train --data config/coco.data --pretrained_weights weights/yolov3.weights --block_pattern_prune --admm_retrain --resume_from /mnt/Data-Weight/1xN_new/yolov3/checkpoint/admm_retrain/4_patterns/yolov3_last.pth --epochs 5