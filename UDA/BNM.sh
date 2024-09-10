export PATH="/home/amirkh/miniconda3/envs/taxonomist/bin:$PATH"

python main.py \
  --backbone resnet152 \
  --transfer_loss_weight 1 \
  --transfer_loss bnm \
  --lr 0.01 \
  --weight_decay 0.00001 \
  --batch_size 32 \
  --momentum 0.9 \
  --lr_scheduler True \
  --lr_gamma 0.0003 \
  --lr_decay 0.75 \
  --n_iter_per_epoch 500 \
  --n_epoch 90 \
  --seed 1 \
  --num_workers 32



# chmod +x BNM.sh
# ./BNM.sh > BNM.txt 2>&1
