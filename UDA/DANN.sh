export PATH="/home/amirkh/miniconda3/envs/taxonomist/bin:$PATH"

python main.py \
  --backbone resnet152 \
  --transfer_loss_weight 1.0 \
  --transfer_loss adv \
  --lr 0.01 \
  --weight_decay 0.01 \
  --batch_size 32 \
  --momentum 0.9 \
  --lr_scheduler True \
  --lr_gamma 0.01 \
  --lr_decay 0.75 \
  --n_iter_per_epoch 200 \
  --n_epoch 90 \
  --seed 1 \
  --num_workers 32

# squeue -l -u $USER
# scancel $(squeue -u $USER -n DANN -o "%.18i" | tail -n 1)
# ./DANN.sh > DANN.txt 2>&1
