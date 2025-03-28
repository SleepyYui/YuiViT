venv/bin/python main.py \
    --mode evaluate \
    --data_dir traindata/dataset \
    --checkpoint traindata/checkpoints/checkpoint.pth \
    --output_dir traindata/evaluation_results \
    --num_workers 16
