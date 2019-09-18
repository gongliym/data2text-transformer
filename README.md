# data2text-transformer
Code for Enhanced Transformer Model for Data-to-Text Generation (Gong, Crego, Senellart; WNGT2019).
Much of this code is adapted from an earlier fork of [XLM](https://github.com/facebookresearch/XLM).

## Dataset and Preprocessing

The boxscore-data json files can be downloaded from the [boxscore-data repo](https://github.com/harvardnlp/boxscore-data).

Assuming the RotoWire json files reside at `./rotowire`, the following commands will preprocess the data

### Step1: Data extraction 

```
python scripts/data_extract.py -d rotowire/train.json -o rotowire/train
```

In this step, we:

* Convert the tables into a sequence of records: `train.gtable`
* Extract the summary and transform entity tokens (such as **Kobe Bryant** -> **Kobe_Bryant**): `train.summary`
* Mark the occurrances of records in the summary: `train.gtable_label` and `train.summary_label`

### Step2: Extract vocabulary

```
python scripts/extract_vocab.py -t rotowire/train.gtable -s rotowire/train.summary
```
It will generate vocabulary files for each of them:

* `rotowire/train.gtable_vocab`
* `rotowire/train.summary_vocab`

### Step3: Binarize the data

```
python model/preprocess_summary_data.py --summary rotowire/train.summary \
                                        --summary_vocab rotowire/train.summary_vocab \
                                        --summary_label rotowire/train.summary_label
                                        
python model/preprocess_table_data.py --table rotowire/train.gtable \
                                      --table_label rotowire/train.gtable_label \
                                      --table_vocab rotowire/train.gtable_vocab
```
And we finally get the training data:
* Input record sequences: `train.gtable.pth`
* Output summaries: `train.summary.pth`

## Model Training
```
MODELPATH=$PWD/model
export PYTHONPATH=$MODELPATH:$PYTHONPATH

python $MODELPATH/train.py

## main parameters
--model_path "experiments"
--exp_name "baseline"
--exp_id "try1"

## data location / training objective
--train_cs_table_path rotowire/train.gtable.pth        # record data for content selection (CS) training
--train_sm_table_path rotowire/train.gtable.pth        # record data for data2text summarization (SM) training
--train_sm_summary_path rotowire/train.summary.pth     # summary data for data2text summarization (SM) training
--valid_table_path rotowire/valid.gtable.pth           # input record data for validation
--valid_summary_path rotowire/valid.summary.pth        # output summary data for validation
--cs_step True                                         # enable content selection training objective
--lambda_cs "1"                                        # CS training coefficient
--sm_step True                                         # enable summarization objective
--lambda_sm "1"                                        # SM training coefficient
    
## transformer parameters
--label_smoothing 0.05                                 # label smoothing
--share_inout_emb True                                 # share the embedding and softmax weights in decoder
--emb_dim 512                                          # embedding size
--enc_n_layers 1                                       # number of encoder layers
--dec_n_layers 6                                       # number of decoder layers
--dropout 0.1                                          # dropout

## optimization
--save_periodic 1                                      # save model every N epoches
--batch_size 6                                         # batch size (number of examples)
--beam_size 4                                          # beam search in generation
--epoch_size 1000                                      # number of examples per epoch
--eval_bleu True                                       # evaluate the BLEU score
--validation_metrics valid_mt_bleu                     # validation metrics
```
