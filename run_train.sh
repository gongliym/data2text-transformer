#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

MODELPATH=$PWD/model
export PYTHONPATH=$MODELPATH:$PYTHONPATH

python $MODELPATH/train.py \
    --model_path "experiments" \
    --exp_name "data2text-transformer" \
    --exp_id "baseline" \
    --train_cs_table_path rotowire/train.gtable.pth \
    --train_sm_table_path rotowire/train.gtable.pth \
    --train_sm_summary_path rotowire/train.summary.pth \
    --valid_table_path rotowire/valid.gtable.pth \
    --valid_summary_path rotowire/valid.summary.pth \
    --cs_step True \
    --lambda_cs "1" \
    --sm_step True \
    --lambda_sm "1" \
    --label_smoothing 0.05 \
    --sm_step_with_cc_loss False \
    --sm_step_with_cs_proba False \
    --share_inout_emb True \
    --share_srctgt_emb False \
    --sinusoidal_embeddings False \
    --emb_dim 512 \
    --enc_n_layers 1 \
    --dec_n_layers 6 \
    --dropout 0.1 \
    --save_periodic 1 \
    --batch_size 6 \
    --beam_size 4 \
    --epoch_size 1000 \
    --eval_bleu True \
    --validation_metrics valid_mt_bleu

