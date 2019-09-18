# data2text-transformer
Code for Enhanced Transformer Model for Data-to-Text Generation (Gong, Crego, Senellart; WNGT2019).
Much of this code is adapted from an earlier fork of [XLM](https://github.com/facebookresearch/XLM).

# Dataset
The boxscore-data json files can be downloaded from the [boxscore-data repo](https://github.com/harvardnlp/boxscore-data).

# Preprocessing

Assuming the RotoWire json files reside at `./rotowire`, the following commands will preprocess the data

## Step1: Data extraction 

```
python scripts/data_extract.py -d rotowire/train.json -o rotowire/train
```

In this step, we:

* Convert the tables into a sequence of records: `train.gtable`
* Extract the summary and transform entity tokens (such as **Kobe Bryant** -> **Kobe_Bryant**): `train.summary`
* Mark the occurrances of records in the summary: `train.gtable_label` and `train.summary_label`

## Step2: Extract vocabulary

```
python scripts/extract_vocab.py -t rotowire/train.gtable -s rotowire/train.summary
```
It will generate vocabulary files for each of them:

* `rotowire/train.gtable_vocab`
* `rotowire/train.summary_vocab`

## Step3: Binarize the data

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
