venv/bin/python NSFWDetect/main.py \
    --mode evaluate \
    --data_dir traindata/dataset \
    --checkpoint traindata/checkpoints/checkpoint.pth \
    --output_dir traindata/evaluation_results \
    --start_tensorboard \
    --tensorboard_port 6007 \
