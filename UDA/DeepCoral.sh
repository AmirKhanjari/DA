export PATH="/home/amirkh/miniconda3/envs/taxonomist/bin:$PATH"

python main.py \
  --backbone resnet18 \
  --transfer_loss_weight 10.0 \
  --transfer_loss coral \
  --lr 3e-3 \
  --weight_decay 5e-4 \
  --batch_size 264 \
  --momentum 0.9 \
  --lr_scheduler True \
  --lr_gamma 0.0003 \
  --lr_decay 0.75 \
  --n_iter_per_epoch 500 \
  --n_epoch 90 \
  --seed 1 \
  --num_workers 32



# chmod +x DeepCoral.sh
# ./DeepCoral.sh > DeepCoral.txt 2>&1
