export PATH="/home/amirkh/miniconda3/envs/taxonomist/bin:$PATH"

CUDA_VISIBLE_DEVICES=0

python main.py \
  --backbone resnet18 \
  --transfer_loss_weight 0.5 \
  --transfer_loss lmmd \
  --lr 0.01 \
  --weight_decay 0.00001 \
  --batch_size 32 \
  --momentum 0.9 \
  --lr_scheduler True \
  --lr_gamma 0.0003 \
  --lr_decay 0.75 \
  --n_iter_per_epoch 500 \
  --n_epoch 32 \
  --seed 1 \
  --num_workers 32



# chmod +x DSAN.sh
# ./DSAN.sh > DSAN.txt 2>&1
