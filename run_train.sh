
python train.py \
    --dataset_name=your_data1 \
    --epochs=50 \

python train.py \
    --dataset_name=your_data2 \
    --epochs=50 \
    --monitor_metric='val_acc' \
    --use_early_stopping=TRUE \
    --monitor_mode='max' \
    --lr_scheduler='None'