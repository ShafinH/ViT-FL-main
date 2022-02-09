### 3. Training
* cd `a_DynConv/' and run 
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train.py \
--train_list='list/MOTS/MOTS_train.txt' \
--snapshot_dir='snapshots/dodnet' \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=2 \
--num_epochs=1000 \
--start_epoch=0 \
--learning_rate=1e-2 \
--num_classes=2 \
--num_workers=8 \
--weight_std=True \
--random_mirror=True \
--random_scale=True \
--FP16=False
```

### 4. Evaluation
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--val_list='list/MOTS/MOTS_test.txt' \
--reload_from_checkpoint=True \
--reload_path='snapshots/dodnet/MOTS_DynConv_checkpoint.pth' \
--save_path='outputs/' \
--input_size='64,192,192' \
--batch_size=1 \
--num_gpus=1 \
--num_workers=2
```

### 5. Post-processing
```
python postp.py --img_folder_path='outputs/dodnet/'
```