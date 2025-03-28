venv/bin/python NSFWDetect/main.py \
    --mode train \
    --data_dir traindata/dataset \
    --batch_size 64 \
    --epochs 50 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --checkpoint_dir traindata/checkpoints \
    --tensorboard_dir traindata/tensorboard \
    --use_mixed_precision \
    --cache_dataset \
    --gpu 0 \
    --output_dir traindata/outputs \
    --start_tensorboard \
    --tensorboard_port 6006 \
    # --checkpoint traindata/checkpoints/checkpoint.pth \
